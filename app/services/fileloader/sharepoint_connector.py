import requests
from msal import ConfidentialClientApplication
import os
from typing import List, Dict, Optional
from urllib.parse import urlparse

class SharePointConnector:
    def __init__(self, tenant_id: str, client_id: str, client_secret: str, site_url: str, supported_extensions: Optional[List[str]] = None):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.site_url = self._format_site_url(site_url)
        self.supported_extensions = supported_extensions if supported_extensions else []
        self.graph_api = "https://graph.microsoft.com/v1.0"
        self.access_token = self._get_access_token()
        self.site_id = self._get_site_id()

    def _format_site_url(self, url: str) -> str:
        """
        Format the SharePoint site URL for both personal and team sites.
        
        Examples:
        Personal: https://dhiai-my.sharepoint.com/personal/user_name -> dhiai-my.sharepoint.com:/personal/user_name
        Team: https://dhiai.sharepoint.com/sites/team-name -> dhiai.sharepoint.com:/sites/team-name
        """
        parsed = urlparse(url)
        hostname = parsed.netloc
        path_parts = parsed.path.strip('/').split('/')
        
        if 'personal' in url:
            # Handle personal sites
            username = path_parts[path_parts.index('personal') + 1]
            return f"{hostname}:/personal/{username}"
        else:
            # Handle team sites
            # Remove empty strings from path parts
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
            raise Exception(f"Failed to get access token: {token_response.get('error_description', 'Unknown error')}")
        return token_response["access_token"]

    def _get_site_id(self) -> str:
        """Get the SharePoint site ID."""
        response = requests.get(
            f"{self.graph_api}/sites/{self.site_url}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get site ID: {response.text}")
        return response.json().get("id")

    def get_files_with_metadata(self, directory_path: str = "") -> List[Dict]:
        """
        Recursively fetch all files and their metadata in the directory and subdirectories.
        
        Args:
            directory_path: The starting directory path (default: root)
            
        Returns:
            List of dictionaries containing file metadata
        """
        files = []
        
        def fetch_items(path: str):
            # Construct the appropriate endpoint URL based on the path
            if not path:
                endpoint = f"{self.graph_api}/sites/{self.site_id}/drive/root/children"
            else:
                # Remove leading slash if present and encode the path
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
                # Construct the item path
                item_path = f"{path}/{item['name']}" if path else f"/{item['name']}"
                item_path = item_path.replace('//', '/')  # Clean up any double slashes
                
                if "folder" in item:
                    # Recursively fetch items in subfolders
                    fetch_items(item_path)
                else:
                    # Check if file extension is supported (if any extensions are specified)
                    if not self.supported_extensions or os.path.splitext(item["name"])[1].lower() in self.supported_extensions:
                        # Construct metadata dictionary
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

    def download_file(self, file_metadata: Dict) -> bytes:
        """Download a file using its metadata."""
        download_url = file_metadata.get("download_url")
        if not download_url:
            raise Exception("Download URL not found in metadata.")
        
        response = requests.get(download_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download file: {response.text}")
        
        return response.content