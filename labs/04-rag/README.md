# RAG examples :

## Creating the Vector DB

1- Open docker (docker desktop for windows)
2- launche the docker compose using 
```bash
docker compose up -d
```
3- Check the Qdrant UI on : localhost:3336/Dashboard 

## Downloading the data

4- Download the files at : https://www.kaggle.com/datasets/saibhossain/rag-practice

5- Create a folder named data under 04-rag : 04-rag/data
6- put the downloaded files inside 04-rag/data

## innitializing the Database

7 - Run 10-init_qdrant.py and 11-qdrant_retreival.py
