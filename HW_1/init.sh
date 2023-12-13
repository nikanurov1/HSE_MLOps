echo 'Start Minio server'
minio server minio/
echo 'Alias minio'
mc alias set s3storage http://127.0.0.1:9000 9WAUcqwOibQuf0vgm9m0 81FmjNY9DUVZzNRNJWSOsUKr9N94t5fKm7GwBCFJ
echo 'Create bucket'
mc mb s3storage/hw-mlops

echo "Init dvc"
dvc init -f
dvc remote remove s3storage
dvc remote add -d s3storage s3://hw-mlops -f
dvc remote modify s3storage endpointurl http://127.0.0.1:9000
dvc remote modify s3storage access_key_id obai9Szm6zF7XpWr6UTQ
dvc remote modify s3storage secret_access_key vNZBQjoCigmI6QDhMIG2BQhm6Vgx4WFGqQkhAdZ4

python api.py