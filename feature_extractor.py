import re
import numpy as np

IP_PATTERN = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
URGENT_KEYWORDS = ["verify","urgent","suspended","act now","confirm","password","click here","limited time","account locked"]


def _name_domain_mismatch(display_name: str, domain: str)->bool:

    if not display_name or not domain:
        return False
    name_words = re.findall(r'[a-zA-Z]{3,}', display_name.lower())
    if not name_words:
        return False
    domain_root= domain.lower().split(".")[0]
    return not any(word in domain_root or domain_root in word for word in name_words)



def extract_structural_features(parsed: dict)->dict:

    urls=parsed.get("urls", [])
    body=parsed.get("body", "")
    sender_domain=parsed.get("sender_domain", "")
    sender_display_name=parsed.get("sender_display_name", "")

    num_urls=len(urls)
    has_ip_url=any(IP_PATTERN.match(u) for u in urls)
    has_at_in_url=any("@" in u for u in urls)

    url_domains=set()

    for u in urls:
        m=re.search(r'https?://([^/]+)/?', u)
        if m:
            url_domains.add(m.group(1).lower())


    sender_domain_mismatch=bool(sender_domain) and bool(sender_domain) and (sender_domain.lower() not in url_domains)

    has_html=bool(re.search(r'<\s*html', body, re.IGNORECASE))
    has_form=bool(re.search(r'<\s*/?\s*form', body, re.IGNORECASE))
    has_iframe=bool(re.search(r'<\s*/?\s*iframe', body, re.IGNORECASE))

    urgent_keyword_count=sum(1 for kw in URGENT_KEYWORDS if kw in body.lower())

    name_domain_mismatch=_name_domain_mismatch(sender_display_name, sender_domain)

    return{
        "num_urls": num_urls,
        "has_ip_url": has_ip_url,
        "has_at_in_url": has_at_in_url,
        "sender_domain_mismatch": sender_domain_mismatch,
        "has_html": has_html,
        "has_form": has_form,
        "has_iframe": has_iframe,
        "urgent_keyword_count": urgent_keyword_count,
        "name_domain_mismatch": name_domain_mismatch
    }

