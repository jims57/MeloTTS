Python API
English with Multiple Accents
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)
speaker_ids = model.hps.data.spk2id

# American accent
output_path = 'en-us.wav'
model.tts_to_file(text, speaker_ids['EN-US'], output_path, speed=speed)

# British accent
output_path = 'en-br.wav'
model.tts_to_file(text, speaker_ids['EN-BR'], output_path, speed=speed)

# Indian accent
output_path = 'en-india.wav'
model.tts_to_file(text, speaker_ids['EN_INDIA'], output_path, speed=speed)

# Australian accent
output_path = 'en-au.wav'
model.tts_to_file(text, speaker_ids['EN-AU'], output_path, speed=speed)

# Default accent
output_path = 'en-default.wav'
model.tts_to_file(text, speaker_ids['EN-Default'], output_path, speed=speed)
Spanish
from melo.api import TTS

# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can also change to cuda:0
device = 'cpu'

text = "El resplandor del sol acaricia las olas, pintando el cielo con una paleta deslumbrante."
model = TTS(language='ES', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'es.wav'
model.tts_to_file(text, speaker_ids['ES'], output_path, speed=speed)
French
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "La lueur dorée du soleil caresse les vagues, peignant le ciel d'une palette éblouissante."
model = TTS(language='FR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'fr.wav'
model.tts_to_file(text, speaker_ids['FR'], output_path, speed=speed)
Chinese
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
model = TTS(language='ZH', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'zh.wav'
model.tts_to_file(text, speaker_ids['ZH'], output_path, speed=speed)
Japanese
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "彼は毎朝ジョギングをして体を健康に保っています。"
model = TTS(language='JP', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'jp.wav'
model.tts_to_file(text, speaker_ids['JP'], output_path, speed=speed)
Korean
from melo.api import TTS

# Speed is adjustable
speed = 1.0
device = 'cpu' # or cuda:0

text = "안녕하세요! 오늘은 날씨가 정말 좋네요."
model = TTS(language='KR', device=device)
speaker_ids = model.hps.data.spk2id

output_path = 'kr.wav'
model.tts_to_file(text, speaker_ids['KR'], output_path, speed=speed)