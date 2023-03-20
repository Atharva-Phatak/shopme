import glob
import os
import urllib.request
from datetime import datetime

import boto3
import pandas as pd
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import (
    PythonOperator,
)
from dotenv import load_dotenv


load_dotenv()


def get_csv_files(bucket, prefix):
    """Get csv files from s3 bucket for particular category"""
    files_gen = bucket.objects.all()
    files = [
        f"{Params.BUCKET}/{fpath.key}"
        for fpath in files_gen
        if (prefix in fpath and ".csv" in fpath)
    ]
    return files


def connect_s3(prefix: str, bucket_name: str):
    """Connect to s3 bucket and get the required csv file"""
    session = boto3.session()
    s3_resource = session.resource("s3")
    bucket = s3_resource.bucket(bucket_name)
    files = get_csv_files(bucket, prefix)
    for file in csv_files:
        frame_response = pd.read_csv(file)
        frames.append(frame_response)
    frame = pd.concat(frames)
    print(
        "Successfully read files from S3-bucket"
    )
    return frame


def store_images(prefix, bucket_name):
    """Store images to local system"""
    frame = connect_s3(prefix, bucket_name)
    ids = [
        x.split(".")[-2]
        for x in frame.product_page_links
    ]
    images = [
        f"https:{x}" for x in frame.image_links
    ]
    for idx, link in zip(ids, images):
        urllib.request.urlretrieve(
            link, f"{Params.ROOTDIR}{idx}.jpg"
        )


# Uncomment below to push images to s3 bucket >> Not recommended. USE DVC instead
# def push_images_to_s3(rootdir, bucket_name):
#    client = boto3.client("s3")
#    for fpath in glob.glob(rootdir):
#        idx = fpath.split("/")[-1]
#        client.upload_file(
#            fpath,
#            bucket_name,
#            f"image_store/{idx}",
#        )


with DAG(
    dag_id="image-collection-ptask",
    start_date=datetime(2023, 1, 29),
    schedule_interval="@once",
    catchup=False,
) as dag:
    baby_task = PythonOperator(
        task_id="baby_image_collection",
        python_callable=store_images,
        op_kwargs={
            "prefix": "baby",
            "bucket_name": os.getenv("BUCKET"),
        },
    )

    kids_task = PythonOperator(
        task_id="kids_image_collection",
        python_callable=store_images,
        op_kwargs={
            "prefix": "kids",
            "bucket_name": os.getenv("BUCKET"),
        },
    )

    women_task = PythonOperator(
        task_id="women_image_collection",
        python_callable=store_images,
        op_kwargs={
            "prefix": "women",
            "bucket_name": os.getenv("BUCKET"),
        },
    )

    men_task = PythonOperator(
        task_id="men_image_collection",
        python_callable=store_images,
        op_kwargs={
            "prefix": "men",
            "bucket_name": os.getenv("BUCKET"),
        },
    )
    task_start = BashOperator(
        task_id="start", bash_command="date"
    )

(
    task_start
    >> [
        baby_task,
        kids_task,
        women_task,
        men_task,
    ]
)
