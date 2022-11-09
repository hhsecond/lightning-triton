import base64
import subprocess
import tarfile
from io import BytesIO
from typing import List
from urllib import request

import lightning as L
import uvicorn
from fastapi import FastAPI
from lightning_api_access import APIAccessFrontend
from pydantic import BaseModel


class DiffusionBuildConfig(L.BuildConfig):
    def build_commands(self) -> List[str]:
        return ["pip install -r triton-requirements.txt"]


class TritonServe(L.LightningWork):
    def __init__(self, cloud_compute: L.CloudCompute):
        super().__init__(
            cloud_compute=cloud_compute,
            cloud_build_config=DiffusionBuildConfig(
                image="ghcr.io/gridai/lightning-triton:v0.9"
            ),
            parallel=True,
        )

    def run(self):
        request.urlretrieve(
            "https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/sd_weights.tar.gz",
            "sd_weights.tar.gz",
        )
        file = tarfile.open("sd_weights.tar.gz")
        file.extractall("model_repository/diffusion/1/stable_diffusion_weights")
        subprocess.run(
            [
                "tritonserver",
                "--model-repository=model_repository",
                "--http-port",
                str(self.port),
            ]
        )


class APIComponent(L.LightningWork):
    def __init__(self):
        super().__init__(cloud_build_config=DiffusionBuildConfig(), parallel=True)

    def run(self, serve_engine_url):
        import numpy as np
        import tritonclient.http as httpclient
        from PIL import Image
        from tritonclient.utils import np_to_triton_dtype

        fastapi_app = FastAPI()

        class Data(BaseModel):
            prompt: str

        @fastapi_app.post("/predict")
        async def predict(data: Data):
            client = httpclient.InferenceServerClient(
                url=serve_engine_url, connection_timeout=1200.0, network_timeout=1200.0
            )
            text_obj = np.array([data.prompt], dtype="object").reshape((-1, 1))
            input_text = httpclient.InferInput(
                "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
            )
            input_text.set_data_from_numpy(text_obj)
            output_img = httpclient.InferRequestedOutput("generated_image")
            query_response = client.infer(
                model_name="diffusion",
                inputs=[input_text],
                outputs=[output_img],
                timeout=240,
            )
            image = Image.fromarray(
                np.squeeze(query_response.as_numpy("generated_image").astype(np.uint8))
            )
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return {"image": f"data:image/png;base64,{img_str}"}

        uvicorn.run(
            fastapi_app,
            host=self.host,
            port=self.port,
            loop="uvloop",
            timeout_keep_alive=60,
        )


class APIUsageFlow(L.LightningFlow):
    def __init__(self, api_url: str = ""):
        super().__init__()
        self.api_url = api_url

    def configure_layout(self):
        return APIAccessFrontend(
            apis=[
                {
                    "name": "Generate Image",
                    "url": f"{self.api_url}/predict",
                    "method": "POST",
                    "request": {"prompt": "cats in hats"},
                    "response": {
                        "image": "data:image/png;base64,<image-actual-content>"
                    },
                }
            ]
        )


class DiffusionServeFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.triton_server = TritonServe(
            cloud_compute=L.CloudCompute("cpu-medium", disk_size=100)
        )
        self.api_component = APIComponent()
        self.api_usage_flow = APIUsageFlow()

    def run(self):
        self.triton_server.run()
        if self.triton_server.is_running and self.triton_server.internal_ip:
            self.api_component.run(
                serve_engine_url=f"{self.triton_server.internal_ip}:{self.triton_server.port}"
            )
        if self.api_component.is_running and self.api_component.url:
            self.api_usage_flow.api_url = self.api_component.url

    def configure_layout(self):
        return [
            {"name": "API", "content": self.api_usage_flow}
        ]


app = L.LightningApp(DiffusionServeFlow())
