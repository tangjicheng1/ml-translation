# How to use
1. You can build a image with app/Dockerfile, or just pull a image which I have built.
```
cd app
docker build -t tang2432/ml-assignment:v1 .
```
or
```
docker pull tang2432/ml-assignment:v1
```

2. After getting a image, you can just run it and test.  
If you want to run it on GPU, you need nvidia-docker.
```
docker run -td --name translation_test -p 9527:9527 tang2432/ml-assignment:v1
# or you want to run it on GPU
nvidia-docker run -td --name translation_test -p 9527:9527 tang2432/ml-assignment:v1
```

3. Testing with example
```bash
curl --location --request POST 'http://127.0.0.1:9527/translation' \
--header 'Content-Type: application/json' \
--data-raw '{
    "payload": {
        "fromLang": "en",
        "records": [
            {
                "id": "123",
                "text": "Life is like a box of chocolates."
            }
        ],
        "toLang": "ja"
    }
}'
```
You will get a response, like this:
```
{ "result":[ { "id":"123", "text":"人生はチョコレートの箱のようなものだ。" } ] }
```

# What did I do

## 1. I optimized inference performance  
Hardware infomation  
| CPU                                | GPU                   | Memory |
|------------------------------------|-----------------------|--------|
| Intel(R) Xeon(R) Silver 40 cores   | NVIDIA Tesla T4 (16G) | 376G   |

### a) Faster inference engine  
I implemented an inference engine based on onnxruntime with 122% faster than pytorch.   
Here is performance statistics.  
Note: Inference cost time only includes the inference part on GPU, not including data copy between CPU and GPU.   
Test Case is the example, English to Japanese: "Life is like a box of chocolates."  

| inference engine | Pytorch | My engine |
|------------------|---------|-----------|
| cost time        | 486ms   | 218ms     |

### b) Dynamic batch inference  
I implemented batch inference for translation.   
Batch inference is faster than excuting multiple single inferences,
but you should have enough GPU memory.  
Here is some performance statistics of my inference engine.   

| batch            | 1     | 8     | 16     |
|------------------|-------|-------|--------|
| cost time        | 218ms | 799ms | 1360ms |
| GPU memory usage | 4309M | 6065M | 7709M  |

## 2. I improved CPU/GPU utilization

### a) Make CPU/GPU run in parallel  
There are 3 part of translation, preprocess/inference/postprocess.  
- Preprocess: text to tensor, excuted on CPU  
- Inference: tensor to tensor, executed on GPU  
- PostProcess:  tensor to text, excuted on CPU  

Based on this, I implemented a software pipeline.   
So that the first request and the second request can be executed in parallel.  
Just like this:  

| Device      | CPU | GPU   | CPU   | GPU  |
|-------------|-----|-------|-------|------|
| 1st request | pre | infer | post  |      |
| 2ed request |     | pre   | infer | post |

As we can see, when 1st request execute infer on GPU,  
at the same time, 2ed request can execute pre on CPU.  

## 3. I built a scalable service  
I built the scalable service with kubernetes.  
One instance need one GPU.  
If you have a multi-GPU system, you can deploy more than one instances.  
By default, you can deploy one instance with k8s/delopyment.yaml.  
```
kubectl apply -f delopyment.yaml
```
If you want to scale instances, you can execute this command to run 4 instances on 4 GPUs.
```
kubectl scale deployment/trans-deployment --replicas=4
```

If you do not have available GPU, and just want to run it on CPU.  
You need modify deployment.yaml, and delete resources limits.
```
        ports:
        - containerPort: 9527
        # resources:
        #   limits:
        #     nvidia.com/gpu: 1
```

### Note:
Image is large (about 20G) and docker hub registry is slow.  
It is easily failed when k8s pull image due to 2 minutes timeout.   
So, I suggest that use "docker pull" to pull this image.  
And store this images on your own docker registry.  
It is necessary to make image smaller,  
but it will take more time to choose smaller base image and pick up required packages.  

## 4. I completed the featrue
### a) Support for multilingual translation requests  
Not only support English to Japanese, it also supports about 100 languages at the same time.  
Just like : 

```
{ "payload": { 
    "fromLang": "en", "records": [ { "id": "456", "text": "Life is like a box of chocolates." } ], 
    "toLang": "zh" }}
```
The results are :

```
{ "result":[ { "id":"456", 
            "text":"生活就像一盒巧克力。" } ] }
```

## 5. I finished this assignment with good practice

- follow Google Style Guide 

- write readable git commit message and code comments 


# What we can do in the futrue
- Use Redis to cache some results, it can reduce repeated computing for the same request.
- Use storage services for model files, because now I just put model files inside the container and it wastes storage space.
- Add logging for intermediate results, it is very helpful for tracking online problems.


# ML Assignment
Please implement a translation inference service that runs on Kubernetes and provides a RESTful API on port 9527.

The translation model is `M2M100`, and the example can be found in `app/translation_example.py`.

You should first fork this repository, and then send us the code or the url of your forked repository via email. 

**Please do not submit any pull requests to this repository.**


## Delivery
- **app/Dockerfile**: To generate an application image
- **k8s/deployment.yaml**: To deploy image to Kubernetes
- Other necessary code

## Input/Output

When you execute this command:
```bash
curl --location --request POST 'http://127.0.0.1:9527/translation' \
--header 'Content-Type: application/json' \
--data-raw '{
    "payload": {
        "fromLang": "en",
        "records": [
            {
                "id": "123",
                "text": "Life is like a box of chocolates."
            }
        ],
        "toLang": "ja"
    }
}'
```

Should return:
```bash
{
   "result":[
      {
         "id":"123",
         "text":"人生はチョコレートの箱のようなものだ。"
      }
   ]
}
```

## Bonus points
- Clean code
- Scalable architecture
- Good inference performance
- Efficient CPU/GPU utilization
