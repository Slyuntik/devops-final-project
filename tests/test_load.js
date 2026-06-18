import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 10,
  duration: '30s',
};

export default function () {
  const payload = JSON.stringify({
    text: 'A scientific text for load testing of the classification model. The study presents new findings in the field of machine learning and neural networks.'
  });
  
  const params = {
    headers: { 'Content-Type': 'application/json' },
  };
  
  const url = __ENV.CI ? 'http://app:8000/predict' : 'http://localhost:8000/predict';
  
  const res = http.post(url, payload, params);
  
  check(res, {
    '✅ status is 200': (r) => r.status === 200,
    '✅ response has quality': (r) => JSON.parse(r.body).hasOwnProperty('quality'),
  });
  
  sleep(1);
}