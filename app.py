"""
app.py -- Entrypoint for running Uvicorn server in Docker and local environments.
Binds to 0.0.0.0:8000 so that Docker port publishing (8000:8000) and Codespaces
port forwarding route external client and browser traffic properly.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
