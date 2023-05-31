# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김용석님
- 리뷰어 : 이하영


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [O] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?
  ```Python
  NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
  D_MODEL = 256 # 인코더와 디코더 내부의 입, 출력의 고정 차원
  NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
  UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
  DROPOUT = 0.1 # 드롭아웃의 비율
  ```
- [X] 3.코드가 에러를 유발할 가능성이 있나요?
  <br>(없음)
- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  ```Python
  def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  # 인코더에서 패딩을 위한 마스크
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더에서 미래의 토큰을 마스크 하기 위해서 사용합니다.
  # 내부적으로 패딩 마스크도 포함되어져 있습니다.
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 두 번째 어텐션 블록에서 인코더의 벡터들을 마스킹
  # 디코더에서 패딩을 위한 마스크
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)
  ```
- [O] 5.코드가 간결한가요?
