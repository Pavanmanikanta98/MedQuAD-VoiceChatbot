import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_medlineplus_topic_summary(query):
    """
    Retrieves a topic summary from MedlinePlus for a given query.
    Returns a brief snippet (up to 200 characters) if found; otherwise, returns an empty string.
    """
    print("working")
    base_search_url = "https://vsearch.nlm.nih.gov/vivisimo/cgi-bin/query-meta"
    params = {
        "v:project": "medlineplus",
        "v:sources": "medlineplus-bundle",
        "query": query
    }
    search_url = base_search_url + "?" + urllib.parse.urlencode(params)
    print("MedlinePlus Search URL:", search_url)
    
    try:
        search_response = requests.get(search_url, timeout=10)
        if search_response.status_code != 200:
            print("Search response error:", search_response.status_code)
            return ""
        
        search_soup = BeautifulSoup(search_response.text, "html.parser")
        results_list = search_soup.find("ol", class_="results")
        if not results_list:
            print("No results found.")
            return ""
        
        first_result = results_list.find("li", class_="document")
        if not first_result:
            print("No document result found.")
            return ""
        
        a_tag = first_result.find("a")
        if not a_tag or not a_tag.get("href"):
            print("No valid link found.")
            return ""
        
        first_link = a_tag.get("href")
        full_link = urllib.parse.urljoin("https://vsearch.nlm.nih.gov", first_link)
        print("First result full link:", full_link)
        
        result_response = requests.get(full_link, timeout=10)
        if result_response.status_code != 200:
            print("Failed to retrieve the first result page.")
            return ""
        
        result_soup = BeautifulSoup(result_response.text, "html.parser")
        topic_summary_div = result_soup.find("div", id="topic-summary", class_="syndicate")
        if topic_summary_div:
            text = topic_summary_div.get_text(separator=" ", strip=True)
            return text[:250]
        else:
            print("Topic summary not found.")
            return ""
    except Exception as e:
        print("Error in get_medlineplus_topic_summary:", e)
        return ""
        
# Example usage for testing:
if __name__ == "__main__":
    query = "diabetes"
    summary = get_medlineplus_topic_summary(query)
    if summary:
        print("Topic Summary from MedlinePlus:")
        print(summary)
    else:
        print("No topic summary could be retrieved.")
