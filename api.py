from fastapi import FastAPI
from pydantic import BaseModel
from main import orchestrator   

app = FastAPI(title="Multi-Agent API", version="1.0")

# Request Model
class UserRequest(BaseModel):
    user_input: str


# GET Method: Home / Health Check
@app.get("/")
async def home():
    return {"message": "API is running successfully!"}


# GET Method: Simple test route
@app.get("/hello")
async def hello():
    return {"message": "Welcome to Multi-Agent API"}


# GET Method: API Info
@app.get("/docs-info")
async def docs_info():
    return {
        "available_endpoints": [
            "GET /",
            "GET /hello",
            "GET /docs-info",
            "POST /echo",
            "POST /run-agents"
        ]
    }


# POST Method: Echo testing input
@app.post("/echo")
async def echo(data: UserRequest):
    return {"you_sent": data.user_input}


# POST Method: Run all agents using orchestrator
@app.post("/run-agents")
async def run_agents(data: UserRequest):
    result = orchestrator(data.user_input)
    return result
