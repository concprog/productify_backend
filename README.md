# Productify (backend)
AI based roadmap-generator (backend)

### Quickstart
First, clone the repo and install deps:  
```sh
git clone https://github.com/nusaturn/openhack_proto_backend
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python
```
Then, install searXNG and configure it to run on `localhost:7120`
```sh
docker pull searxng/searxng
export PORT=7120
docker run --rm \
             -d -p ${PORT}:8080 \
             -v "${PWD}/searxng:/etc/searxng" \
             -e "BASE_URL=http://localhost:$PORT/" \
             -e "INSTANCE_NAME=my-instance" \
             searxng/searxng
```

When you're done, stop SearXNG using `docker container stop {container_id} `
