## Code Peer Review Template
---
* 코더 : 김용석
* 리뷰어 : 정연준


## PRT(PeerReviewTemplate)
---
- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

```python
print("1)원문(전처리전) :", summa_data.iloc[36928].text)
print("\n")
print("2)원문(전처리후) :", seq2text(encoder_input_test[52]))
print("\n")
print("3)실제 요약 :", seq2summary(decoder_input_test[52]))
print("\n")
print("4)추상적 요약 :", decode_sequence(encoder_input_test[52].reshape(1, text_max_len)))
print("\n")
print("5)추출적 요약 :", summarize(summa_data.iloc[36928].text, ratio=0.5))
print("\n")
```

결과물로 1.원문 2.원문을 전처리한 결과 3.실제요약 4.추상적요약 결과 5.추출적요약 결과를 출력하고,
서로 비교할 수 있도록 결과 코드를 작성하였다.

- [x] 주석을 보고 작성자의 코드가 이해되었나요?

설명이 필요한 부분의 경우 주석을 달아 어떤 의미인지 알 수 있었다.

- [x] 코드가 에러를 유발할 가능성이 있나요?

```python
data[data['headlines'] == str_test].index
```

str_test].index  <-- 대괄호 부분 오타인것 같습니다...

- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
```python
src_vocab = 20000
src_tokenizer = Tokenizer(num_words = src_vocab) 
src_tokenizer.fit_on_texts(encoder_input_train) 

tar_vocab = 10000
tar_tokenizer = Tokenizer(num_words = tar_vocab) 
tar_tokenizer.fit_on_texts(decoder_input_train)
tar_tokenizer.fit_on_texts(decoder_target_train)

```
src_vocab = 20000
tar_vocab = 10000

과제 내에서 희귀 단어 수를 제외한 후 남은 단어의 숫자를 기준으로 하여 단어 사전의 크기를 설정하였다.


- [x] 코드가 간결한가요?

더 간결하게 만들 방법을 찾지 못하였습니다.


