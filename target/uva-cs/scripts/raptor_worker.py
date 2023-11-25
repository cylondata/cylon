
import time
import random

import radical.pilot as rp

RANKS = 2


# ------------------------------------------------------------------------------
#
class MyWorker(rp.raptor.MPIWorker):
    '''
    This class provides the required functionality to execute work requests.
    In this simple example, the worker only implements a single call: `hello`.
    '''

    # --------------------------------------------------------------------------
    #
    def __init__(self, cfg):

        super().__init__(cfg)

        self.register_mode('foo', self._dispatch_foo)


    # --------------------------------------------------------------------------
    #
    def _dispatch_foo(self, task):

        import pprint
        self._log.debug('==== running foo\n%s',
                pprint.pformat(task['description']))

        return 'out', 'err', 0, None, None


    # --------------------------------------------------------------------------
    #

# ------------------------------------------------------------------------------

