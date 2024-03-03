# Productify (backend)
AI based roadmap-generator (backend)
  - Uses web search and a Chain-of-Thought prompt to find out how to achieve a goal, given any goal
  - Creates a series of SMART goals that can be completed to achieve the end goal
  - Can be used for short term planning as well.

### Quickstart
First, clone the repo and install deps:  
```sh
git clone https://github.com/nusaturn/openhack_proto_backend
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install llama-cpp-python
```
Download a model file (such as one from [here](https://huggingface.co/TheBloke/dolphin-2.6-mistral-7B-dpo-GGUF))
and edit `settings.yaml` to include the model path.  

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


You might want to adjust the given settings based on your hardware.
