import argparse
from contextlib import asynccontextmanager
from uuid import uuid4

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import VLLMConverter
from pydantic import BaseModel, Field


class Session(BaseModel):
    base_url: str
    sampling_params: dict = Field(default_factory=dict)
    model: str = ""


def create_app() -> FastAPI:
    sessions: dict[str, Session] = {}
    converter = VLLMConverter(return_token_id_information=False, uses_reasoning_parser=True)

    @asynccontextmanager
    async def lifespan(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/sessions")
    async def register(body: Session):
        if "/sessions/" not in body.base_url or not body.base_url.startswith("http://"):
            raise HTTPException(422, "Expected a Miles internal session URL")
        async with app.state.client.get(body.base_url.rstrip("/") + "/v1/models", allow_redirects=False) as response:
            if response.status != 200:
                raise HTTPException(502, "Miles policy model discovery failed")
            body.model = (await response.json())["data"][0]["id"]
        rollout_id = uuid4().hex
        sessions[rollout_id] = body
        return {"rollout_id": rollout_id}

    @app.delete("/sessions/{rollout_id}")
    async def unregister(rollout_id: str):
        sessions.pop(rollout_id, None)
        return {"status": "closed"}

    async def forward(rollout_id: str, body: dict):
        session = sessions.get(rollout_id)
        if session is None:
            raise HTTPException(404, "Unknown or closed Miles rollout session")
        # Only policy calls reach this bridge. The fixed user LLM has its own model server.
        payload = {**body, **session.sampling_params, "stream": False, "model": session.model}
        async with app.state.client.post(
            session.base_url.rstrip("/") + "/v1/chat/completions", json=payload, allow_redirects=False
        ) as response:
            if response.status != 200:
                raise HTTPException(502, "Miles policy session request failed")
            return await response.json()

    @app.post("/ng-rollout/{rollout_id}/v1/chat/completions")
    async def chat(rollout_id: str, body: dict):
        return await forward(rollout_id, body)

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, body: NeMoGymResponseCreateParamsNonStreaming):
        chat_body = converter.responses_to_chat_completion_create_params(body)
        result = await forward(rollout_id, chat_body.model_dump(exclude_none=True))
        completion = NeMoGymChatCompletion.model_validate(result)
        body.model = completion.model
        return converter.chat_completion_to_response(responses_create_params=body, chat_completion=completion)

    return app


def main():
    parser = argparse.ArgumentParser(description="Route NeMo Gym policy calls through Miles token-tracing sessions")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
