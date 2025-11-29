"""
Local API Testing Script
Tests both /extract-bill-data and /api/v1/hackrx/run endpoints
"""

import json
import sys
import time
from typing import Optional
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


# ============================================================================
# Configuration
# ============================================================================

BASE_URL = "http://localhost:8000"
TIMEOUT = 60  # seconds
REQUEST_DELAY = 1  # seconds between requests

# Sample test URLs (replace with actual Datathon URLs)
SAMPLE_URLS = [
    "https://example.com/sample-bill-1.pdf",
    "https://example.com/sample-bill-2.pdf",
    # Add more test URLs here
]


# ============================================================================
# Helper Functions
# ============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_json(data: dict, indent: int = 2):
    """Print JSON data in a formatted way."""
    try:
        formatted = json.dumps(data, indent=indent, ensure_ascii=False)
        print(formatted)
    except Exception as e:
        print(f"Error formatting JSON: {e}")
        print(data)


def test_endpoint(
    endpoint: str,
    payload: dict,
    description: str
) -> Optional[dict]:
    """
    Test an API endpoint and return the response.
    
    Args:
        endpoint: API endpoint path (e.g., "/extract-bill-data")
        payload: Request payload
        description: Description of what's being tested
        
    Returns:
        Response JSON as dict, or None if request failed
    """
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n📤 Testing: {description}")
    print(f"   Endpoint: {endpoint}")
    print(f"   URL: {url}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            url,
            json=payload,
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Response received in {elapsed_time:.2f}s")
        print(f"   Status Code: {response.status_code}")
        
        # Check if response is successful
        if response.status_code == 200:
            try:
                response_data = response.json()
                print("\n📥 Response JSON:")
                print_json(response_data)
                return response_data
            except json.JSONDecodeError as e:
                print(f"\n❌ Failed to parse JSON response: {e}")
                print(f"   Raw response: {response.text[:500]}")
                return None
        else:
            print(f"\n❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print("   Error details:")
                print_json(error_data)
            except:
                print(f"   Error message: {response.text[:500]}")
            return None
            
    except Timeout:
        print(f"\n❌ Request timed out after {TIMEOUT} seconds")
        return None
    except ConnectionError:
        print(f"\n❌ Connection error: Could not reach {BASE_URL}")
        print("   Make sure the API server is running:")
        print("   uvicorn src.api:app --host 0.0.0.0 --port 8000")
        return None
    except RequestException as e:
        print(f"\n❌ Request failed: {e}")
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None


def test_health_check() -> bool:
    """Test the health check endpoint."""
    print_section("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API is running")
            print_json(data)
            return True
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


# ============================================================================
# Main Testing Functions
# ============================================================================

def test_extract_bill_data_endpoint(test_url: str):
    """Test the /extract-bill-data endpoint."""
    print_section("Testing /extract-bill-data Endpoint")
    
    payload = {"document": test_url}
    response = test_endpoint(
        endpoint="/extract-bill-data",
        payload=payload,
        description="Bill Extraction (Legacy Endpoint)"
    )
    
    if response:
        # Validate response structure
        if "is_success" in response:
            print("\n✅ Response structure is valid")
        else:
            print("\n⚠️  Response structure may be incomplete")
    
    return response


def test_hackrx_endpoint(test_url: str):
    """Test the /api/v1/hackrx/run endpoint."""
    print_section("Testing /api/v1/hackrx/run Endpoint")
    
    payload = {"document": test_url}
    response = test_endpoint(
        endpoint="/api/v1/hackrx/run",
        payload=payload,
        description="HackRx Webhook Endpoint"
    )
    
    if response:
        # Validate HackRx response schema
        required_fields = ["is_success", "token_usage", "data"]
        missing_fields = [field for field in required_fields if field not in response]
        
        if not missing_fields:
            print("\n✅ Response matches HackRx schema")
            
            # Validate data structure
            if "data" in response:
                data = response["data"]
                data_fields = ["pagewise_line_items", "total_item_count", "reconciled_amount"]
                missing_data_fields = [f for f in data_fields if f not in data]
                
                if not missing_data_fields:
                    print("✅ Data structure is complete")
                    print(f"\n📊 Summary:")
                    print(f"   Total pages: {len(data.get('pagewise_line_items', []))}")
                    print(f"   Total items: {data.get('total_item_count', 0)}")
                    print(f"   Reconciled amount: ₹{data.get('reconciled_amount', 0):.2f}")
                else:
                    print(f"⚠️  Missing data fields: {missing_data_fields}")
            else:
                print("⚠️  Missing 'data' field in response")
        else:
            print(f"⚠️  Missing required fields: {missing_fields}")
    
    return response


def run_all_tests():
    """Run all tests with sample URLs."""
    print_section("API Testing Suite")
    print(f"Base URL: {BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print(f"Test URLs: {len(SAMPLE_URLS)}")
    
    # First, check if API is running
    if not test_health_check():
        print("\n❌ API is not running. Please start the server first:")
        print("   uvicorn src.api:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    
    # Test with each sample URL
    results = {
        "extract_bill_data": [],
        "hackrx": [],
        "errors": []
    }
    
    for i, test_url in enumerate(SAMPLE_URLS, 1):
        print_section(f"Test {i}/{len(SAMPLE_URLS)}: {test_url}")
        
        # Test /extract-bill-data endpoint
        result1 = test_extract_bill_data_endpoint(test_url)
        results["extract_bill_data"].append({
            "url": test_url,
            "success": result1 is not None,
            "response": result1
        })
        
        time.sleep(REQUEST_DELAY)  # Avoid overwhelming the server
        
        # Test /api/v1/hackrx/run endpoint
        result2 = test_hackrx_endpoint(test_url)
        results["hackrx"].append({
            "url": test_url,
            "success": result2 is not None,
            "response": result2
        })
        
        time.sleep(REQUEST_DELAY)
    
    # Print summary
    print_section("Test Summary")
    
    extract_success = sum(1 for r in results["extract_bill_data"] if r["success"])
    hackrx_success = sum(1 for r in results["hackrx"] if r["success"])
    
    print(f"\n/extract-bill-data: {extract_success}/{len(SAMPLE_URLS)} successful")
    print(f"/api/v1/hackrx/run: {hackrx_success}/{len(SAMPLE_URLS)} successful")
    
    if extract_success == len(SAMPLE_URLS) and hackrx_success == len(SAMPLE_URLS):
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Bill Extraction API")
    parser.add_argument(
        "--url",
        type=str,
        help="Single URL to test (overrides sample URLs)"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["extract-bill-data", "hackrx", "both"],
        default="both",
        help="Which endpoint(s) to test"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help=f"Base URL of the API (default: {BASE_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT,
        help=f"Request timeout in seconds (default: {TIMEOUT})"
    )
    
    args = parser.parse_args()
    
    # Update configuration from arguments
    BASE_URL = args.base_url
    TIMEOUT = args.timeout
    
    # Determine test URLs
    if args.url:
        test_urls = [args.url]
    else:
        test_urls = SAMPLE_URLS
    
    if not test_urls:
        print("❌ No test URLs provided. Please add URLs to SAMPLE_URLS or use --url flag.")
        sys.exit(1)
    
    # Run tests based on endpoint selection
    if args.endpoint == "both":
        # Run all tests
        run_all_tests()
    elif args.endpoint == "extract-bill-data":
        print_section("Testing /extract-bill-data Only")
        if not test_health_check():
            sys.exit(1)
        for url in test_urls:
            test_extract_bill_data_endpoint(url)
            time.sleep(REQUEST_DELAY)
    elif args.endpoint == "hackrx":
        print_section("Testing /api/v1/hackrx/run Only")
        if not test_health_check():
            sys.exit(1)
        for url in test_urls:
            test_hackrx_endpoint(url)
            time.sleep(REQUEST_DELAY)



