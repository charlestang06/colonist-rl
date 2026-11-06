import hashlib,json,unittest
from pathlib import Path
class ExactPublicFixture(unittest.TestCase):
 def test_original_canonical_observation_hash_is_preserved(self):
  f=Path(__file__).with_name('fixtures') / 'train018-seat0-own95-before.json'
  raw=json.dumps(json.loads(f.read_text()),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
  self.assertEqual(hashlib.sha256(raw).hexdigest(),'b00b0d20011804dc68e1a927a08320f55e5b61b936d0ce28cd4a53da22766ea2')
if __name__=='__main__':unittest.main(verbosity=2)
