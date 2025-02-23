from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import text
from dotenv import load_dotenv
import os
Base = declarative_base()

class DatabaseConnector:
    def __init__(self):
        """
        Initialize the database connection using environment variables.
        """
        database_url = self._create_database_url()
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)

    def _create_database_url(self):
        """
        Read database connection details from environment variables and construct the database URL.

        :return: The connection string for the PostgreSQL database.
        """
        load_dotenv(override=True)  # Load environment variables from .env file

        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASS")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        if not all([db_user, db_password, db_host, db_port, db_name]):
            raise ValueError("Missing one or more database environment variables.")

        return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def get_disclosures(self, number=10, page_filter=None):
        """
        Fetch disclosures with a dynamic page number filter.

        :param number: Number of results to fetch (default is 10).
        :param page_filter: A tuple (min_pages, max_pages) to filter disclosures by page count.
                            Example:
                            - (None, 20) -> pages < 20
                            - (20, 50)   -> pages between 20 and 50
                            - (100, None) -> pages > 100
        :return: List of [disclosure_id, document_url].
        """

        # Base query
        query_str = """
            SELECT d.disclosure_id, dd.document_url
            FROM disclosure.disclosure d
            JOIN disclosure.disclosure_header dh ON d.disclosure_id = dh.disclosure_id
            JOIN disclosure.disclosure_document dd ON d.disclosure_id = dd.disclosure_id
            WHERE 1=1
        """

        # Add dynamic page filter
        params = {"limit": number}
        if page_filter:
            min_pages, max_pages = page_filter

            if min_pages is not None:
                query_str += " AND dh.npages >= :min_pages"
                params["min_pages"] = min_pages

            if max_pages is not None:
                query_str += " AND dh.npages <= :max_pages"
                params["max_pages"] = max_pages

        # Add limit clause
        query_str += " LIMIT :limit"
        
        query = text(query_str)

        with self.Session() as session:
            results = session.execute(query, params).fetchall()
            return [[row.disclosure_id, row.document_url] for row in results]