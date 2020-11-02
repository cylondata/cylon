##
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##


'''
Adopted From: https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py
'''

import urllib
from tqdm.auto import tqdm
from urllib.request import urlretrieve
from urllib.request import urlopen


class TqdmUpTo(tqdm):
    """Alternative Class-based version of the above.
    Provides `update_to(n)` which uses `tqdm.update(delta_n)`.
    Inspired by [twine#242](https://github.com/pypa/twine/pull/242),
    [here](https://github.com/pypa/twine/commit/42e55e06).
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)  # also sets self.n = b * bsize


def download(url, output_file):
    eg_file = url.replace('/', ' ').split()[-1]

    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                  desc=eg_file) as t:  # all optional kwargs
        urlretrieve(url, filename=output_file, reporthook=t.update_to,
                    data=None)
        t.total = t.n

    # Even simpler progress by wrapping the output file's `write()`
    with tqdm.wrapattr(open(output_file, "wb"), "write",
                       miniters=1, desc=eg_file) as fout:
        for chunk in urlopen(url):
            fout.write(chunk)




