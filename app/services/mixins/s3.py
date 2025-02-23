import os
import datetime
import pandas as pd
import boto3
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(override=True)

__all__ = ["S3Connector"]


def format_date(input_date, input_format="%Y-%m-%d", return_format="%Y-%m-%d %H:%M:%S"):
    if isinstance(input_date, pd.Series):
        if not isinstance(input_date[0], pd.Timestamp):
            input_date = pd.to_datetime(input_date, format=input_format)
        input_date = input_date.dt.strftime(return_format)
        return input_date

    if isinstance(input_date, str):
        datetime_value = datetime.datetime.strptime(input_date, input_format)
        datetime_value = datetime_value.strftime(return_format)
        return datetime_value
    else:
        input_date = input_date.strftime(return_format)
        return input_date


class S3Connector:

    @staticmethod
    def get_client():
        return boto3.resource(
            service_name="s3",
            region_name=os.environ.get("AWS_REGION"),
            aws_access_key_id=os.environ.get("aws_access_key_id"),
            aws_secret_access_key=os.environ.get("aws_secret_access_key"),
        )

    def get_pdf_from_s3(self, pdf_path):
        pdf_content = None
        s3_client = self.get_client()
        pdf_file = s3_client.Object(os.environ.get("S3_BUCKET"), pdf_path).get()
        pdf_content = pdf_file["Body"].read()

        return BytesIO(pdf_content)

    def load_file_content(self, bucket_name, file_key):
        s3_client = self.get_client()
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        content = response["Body"].read().decode("utf-8")
        return content
