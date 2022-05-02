import unittest
import pdb

from encoders import ordinal_encoder

class TestEncoders(unittest.TestCase):

    def test_ordinal_encoder(self):
        x = list('aaccbbbdddd')
        encoder = ordinal_encoder(x)
        x_enc_real = [0, 0, 2, 2, 1, 1, 1, 3, 3, 3, 3]

        self.assertEqual(encoder(x), x_enc_real)

    def test_ordinal_encoder_top_n(self):
        x = list('aaccbbbdddd')
        encoder = ordinal_encoder(x, top_n=3)
        x_enc_real = [0, 0, -1, -1, 1, 1, 1, 3, 3, 3, 3]

        self.assertEqual(encoder(x), x_enc_real)


if __name__ == '__main__':
    unittest.main()

