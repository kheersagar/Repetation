import requests

url = 'http://localhost:5000/recommend'
data = {
    'question': 'If Raman drives his bike at a speed of 24 km/h, he reaches his office 5 minutes late. If he drives at a speed of 30 km/h, he reaches his office 4 minutes early. How much time (in minutes) will he take to reach his office at a speed of 27 km/h??',
    'n': 5
}
response = requests.post(url, json=data)

if response.ok:
    recommended_questions = response.json()
    print(recommended_questions)
else:
    print('Error:', response.status_code, response.text)
