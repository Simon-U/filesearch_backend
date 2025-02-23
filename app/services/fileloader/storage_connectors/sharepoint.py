import requests
from msal import ConfidentialClientApplication
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse

from .base import StorageConnector
from ..models import FileLocation, FileMetadata, SharePointConfig, UserInfo, StorageType
from ..exceptions import StorageNotConfiguredError


class SharePointStorageConnector(StorageConnector):
    """
    SharePoint storage connector that handles both authentication and file operations.
    Implements the StorageConnector interface while providing additional SharePoint-specific
    functionality.
    """
    def __init__(self, config: SharePointConfig):
        self.tenant_id = config.tenant_id
        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self.site_url = self._format_site_url(config.site_url)
        self.graph_api = "https://graph.microsoft.com/v1.0"
        self.access_token = self._get_access_token()
        self.site_id = self._get_site_id()

    def _format_site_url(self, url: str) -> str:
        """Format the SharePoint site URL for both personal and team sites."""
        parsed = urlparse(url)
        hostname = parsed.netloc
        path_parts = parsed.path.strip('/').split('/')
        
        if 'personal' in url:
            # Handle personal sites
            username = path_parts[path_parts.index('personal') + 1]
            return f"{hostname}:/personal/{username}"
        else:
            # Handle team sites
            clean_path_parts = [part for part in path_parts if part]
            return f"{hostname}:/{'/'.join(clean_path_parts)}"

    def _get_access_token(self) -> str:
        """Authenticate with Azure AD and get an access token."""
        authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        app = ConfidentialClientApplication(
            self.client_id, 
            authority=authority,
            client_credential=self.client_secret
        )
        token_response = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        
        if "access_token" not in token_response:
            raise StorageNotConfiguredError(
                f"Failed to get SharePoint access token: {token_response.get('error_description', 'Unknown error')}"
            )
        return token_response["access_token"]

    def _get_site_id(self) -> str:
        """Get the SharePoint site ID."""
        response = requests.get(
            f"{self.graph_api}/sites/{self.site_url}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        if response.status_code != 200:
            raise StorageNotConfiguredError(f"Failed to get SharePoint site ID: {response.text}")
        return response.json().get("id")

    def list_files(self, directory_path: str = "") -> List[Dict]:
        """
        List all files and their metadata in the directory and subdirectories.
        This is a SharePoint-specific method that provides richer metadata than the base interface.
        """
        files = []
        
        def fetch_items(path: str):
            # Construct the appropriate endpoint URL based on the path
            if not path:
                endpoint = f"{self.graph_api}/sites/{self.site_id}/drive/root/children"
            else:
                clean_path = path.lstrip('/')
                endpoint = f"{self.graph_api}/sites/{self.site_id}/drive/root:/{clean_path}:/children"

            response = requests.get(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Accept": "application/json",
                    "Prefer": "respond-async"
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to fetch items: {response.text}")

            items = response.json().get("value", [])
            
            for item in items:
                item_path = f"{path}/{item['name']}" if path else f"/{item['name']}"
                item_path = item_path.replace('//', '/')  # Clean up any double slashes
                
                if "folder" in item:
                    fetch_items(item_path)
                else:
                    file_metadata = {
                        "name": item.get("name"),
                        "path": item_path,
                        "size": item.get("size"),
                        "created_datetime": item.get("createdDateTime"),
                        "modified_datetime": item.get("lastModifiedDateTime"),
                        "created_by": item.get("createdBy", {}).get("user", {}),
                        "modified_by": item.get("lastModifiedBy", {}).get("user", {}),
                        "web_url": item.get("webUrl"),
                        "download_url": item.get("@microsoft.graph.downloadUrl"),
                        "file_type": item.get("file", {}).get("mimeType"),
                        "id": item.get("id")
                    }
                    files.append(file_metadata)
        
        fetch_items(directory_path)
        return files

    def get_file_metadata(self, path: str) -> Dict:
        """Get metadata for a specific file."""
        response = requests.get(
            f"{self.graph_api}/sites/{self.site_id}/drive/root:{path}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get file metadata: {response.text}")
        return response.json()

    # Implementation of StorageConnector interface methods
    def get_file_content(self, location: FileLocation) -> bytes:
        """
        Implement the StorageConnector interface method to get file content.
        
        Args:
            location: FileLocation object containing the file path
            
        Returns:
            bytes: The file content
        """
        metadata = self.get_file_metadata(location.path)
        download_url = metadata.get("@microsoft.graph.downloadUrl")
        
        if not download_url:
            raise Exception("Download URL not found in metadata")
        
        response = requests.get(download_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.text}")
        
        return response.content

    def get_metadata(self, location: FileLocation) -> FileMetadata:
        """
        Implement the StorageConnector interface method to get standardized metadata.
        
        Args:
            location: FileLocation object containing the file path
            
        Returns:
            FileMetadata: Standardized metadata object
        """
        sp_metadata = self.get_file_metadata(location.path)
        
        # Extract user information
        created_by = UserInfo(
            name=sp_metadata.get("createdBy", {}).get("user", {}).get("displayName"),
            email=sp_metadata.get("createdBy", {}).get("user", {}).get("email"),
            system_id=sp_metadata.get("createdBy", {}).get("user", {}).get("id")
        )
        
        modified_by = UserInfo(
            name=sp_metadata.get("lastModifiedBy", {}).get("user", {}).get("displayName"),
            email=sp_metadata.get("lastModifiedBy", {}).get("user", {}).get("email"),
            system_id=sp_metadata.get("lastModifiedBy", {}).get("user", {}).get("id")
        )
        
        return FileMetadata(
            file_path=location.path,
            file_type=os.path.splitext(location.path)[1],
            content_type=sp_metadata.get("file", {}).get("mimeType"),
            size_bytes=sp_metadata.get("size", 0),
            universal_url=sp_metadata.get("webUrl"),
            storage_type=StorageType.SHAREPOINT,
            created_at=sp_metadata.get("createdDateTime"),
            modified_at=sp_metadata.get("lastModifiedDateTime"),
            created_by=created_by,
            modified_by=modified_by,
            additional_info={
                "sharepoint_id": sp_metadata.get("id"),
                "download_url": sp_metadata.get("@microsoft.graph.downloadUrl"),
            }
        )