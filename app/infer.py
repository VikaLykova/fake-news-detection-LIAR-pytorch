import sys
import requests

URL = "http://127.0.0.1:8000/analyze"


def main():
    text = " ".join(sys.argv[1:]) or "Мер міста оголосив про безкоштовний проїзд у метро до кінця року"
    resp = requests.post(URL, json={"news_text": text})
    print("Status:", resp.status_code)
    try:
        print(resp.json())
    except Exception:
        print(resp.text)

if __name__ == "__main__":
    main()
