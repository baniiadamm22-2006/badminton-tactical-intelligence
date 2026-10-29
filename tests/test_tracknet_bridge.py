import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tracknet_bridge import infer


class InferenceAdapterTests(unittest.TestCase):
    def test_no_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'predict.py').write_text('')
            py = root / 'python.exe'
            py.write_text('')
            video = root / 'input.mp4'
            video.write_text('')
            with self.assertRaises(FileNotFoundError):
                infer(repo, py, video, root / 'out', root / 'missing.pt', None, 2, False)

    @patch('tracknet_bridge.subprocess.run')
    def test_command_and_output(self, mocked_run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'predict.py').write_text('')
            py = root / 'python.exe'
            py.write_text('')
            video = root / 'input.mp4'
            video.write_text('')
            ckpt = root / 'model.pt'
            ckpt.write_text('')
            output = root / 'out'
            output.mkdir()
            expected = output / 'input_ball.csv'
            expected.write_text('Frame,Visibility,X,Y\n0,0,0,0\n')
            self.assertTrue(infer(repo, py, video, output, ckpt, None, 2, True).samefile(expected))
            command = mocked_run.call_args.args[0]
            self.assertIn('--large_video', command)
            self.assertNotIn('--inpaintnet_file', command)
            self.assertTrue(mocked_run.call_args.kwargs['check'])


if __name__ == '__main__':
    unittest.main()
