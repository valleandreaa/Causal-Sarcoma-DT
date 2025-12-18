# Building the Docker Container

To build the Docker container, run the following command:
For CPU 
   ```sh
   sudo docker build --build-arg DEVICE_TYPE=cpu --no-cache -t models_sarcoma_dt .
   ```
For GPU
   ```sh
   sudo docker build --build-arg DEVICE_TYPE=gpu --no-cache -t models_sarcoma_dt .
   ```
# Start container 
   ```sh
   docker compose up -d
   ```

In case of probems use suda

# Run model training 
```sh
docker exec -it model_training python your_main_script.py --train --model <model_name>
```

You can run this command multiple times for different models using the same container.

# Enter the contianer
```sh
docker exec -it model_training /bin/bash
```
# Restore MongoDB Manually (if necessary)
```sh
docker exec -it mongodb mongorestore --db SarcomaDB /backup/dump/SarcomaDB

```

# Troubleshooting

If you encounter issues, check the logs using:

```sh
docker compose logs
```

To access a running container for debugging purposes, use:

```sh
docker exec -it model_training /bin/bash
```

# Load db

```sh
mongorestore --port 27017 -u root -p example --authenticationDatabase admin  --nsInclude=SarcomaDB.* /dump
```


# Run routines
```sh
chmod +x Models/scripts/train_CGAN_v1_1.sh
```

```sh
Models/scripts/train_CGAN_v1_1.sh
```