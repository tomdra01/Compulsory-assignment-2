import requests


def search_research_papers(topic: str, year: str = None, min_citations: int = 0) -> str:
    """
    Searches for research papers using the OpenAlex API (No API Key required).

    Args:
        topic (str): The research topic.
        year (str, optional): The specific year (e.g., "2023").
        min_citations (int, optional): Minimum number of citations required.

    Returns:
        str: A formatted string containing the top paper results.
    """
    print(
        f"\n[Tool] Searching for papers with TITLE containing '{topic}' (Year: {year}, Min Citations: {min_citations})...")

    base_url = "https://api.openalex.org/works"

    # Use 'title.search' instead of 'default.search' to ensure the paper is actually about the topic
    filter_query = f"title.search:{topic}"

    if year:
        filter_query += f",publication_year:{year}"

    params = {
        "filter": filter_query,
        "sort": "cited_by_count:desc",  # Sort by most cited
        "per_page": 15
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)

        if response.status_code != 200:
            return f"Error: API returned status code {response.status_code}"

        data = response.json()
        results = data.get("results", [])

        if not results:
            return "No papers found matching the criteria."

        valid_papers = []
        for paper in results:
            citations = paper.get("cited_by_count", 0)

            if citations >= min_citations:
                title = paper.get("title", "No Title")
                pub_year = paper.get("publication_year", "N/A")
                link = paper.get("primary_location", {}).get("landing_page_url") or paper.get("id")

                valid_papers.append(
                    f"Title: {title}\n"
                    f"Year: {pub_year}\n"
                    f"Citations: {citations}\n"
                    f"URL: {link}\n"
                    f"---"
                )

        if not valid_papers:
            return f"Found papers on '{topic}', but none had >= {min_citations} citations."

        return "\n".join(valid_papers[:3])

    except Exception as e:
        return f"Error searching for papers: {str(e)}"


# --- Test Block ---
if __name__ == "__main__":
    # Test to see if we get actual CS papers now
    result = search_research_papers(topic="Multi-Agent Systems", year="2023", min_citations=10)
    print(result)