import argparse, requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument('text', nargs='*', help='news text')
    p.add_argument('--url', default='http://127.0.0.1:8000/analyze')
    a = p.parse_args()
    text = " ".join(a.text) or "Мер анонсував безкоштовний проїзд у метро"
    r = requests.post(a.url, json={'news_text': text})
    print("Status:", r.status_code)
    print(r.text)

if __name__ == "__main__":
    main()
