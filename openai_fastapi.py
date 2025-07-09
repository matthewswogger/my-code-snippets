from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from typing import Literal, AsyncGenerator
from openai import AsyncOpenAI
import logging
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)


class Message(BaseModel):
    role: Literal['developer', 'user']
    content: str

class InferenceRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False


async def async_generator(stream) -> AsyncGenerator:
    async for chunk in stream:
        logging.debug(f'Received chunk: {chunk}')

        yield f'data: {json.dumps(chunk.model_dump())}\n\n'


app = FastAPI()
async_client = AsyncOpenAI()


@app.get('/')
async def index():
    return {'foo': 'bar'}


@app.post('/chat/completions')
async def openai_streaming(request: InferenceRequest):
    try:
        chat_completion = await async_client.chat.completions.create(**request.model_dump())

        if not request.stream:
            return chat_completion.model_dump()
        else:
            return StreamingResponse(async_generator(stream=chat_completion))

    except Exception as e:
        logging.error(f'Error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
