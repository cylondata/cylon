#!/usr/bin/env python3

import os
import sys
import time

from collections import defaultdict

import radical.utils as ru
import radical.pilot as rp


def out(msg):
    sys.stdout.write('==== %s\n' % msg)
    sys.stdout.flush()


# This script has to run as a task within a pilot allocation, and is
# a demonstration of a task overlay within the RCT framework. It is expected
# to be staged and launched by the `raptor.py` script in the radical.pilot
# examples/misc directory.
# This master task will:
#
#   - create a master which bootstraps a specific communication layer
#   - insert n workers into the pilot (again as a task)
#   - perform RPC handshake with those workers
#   - send RPC requests to the workers
#   - terminate the worker
#
# The worker itself is an external program which is not covered in this code.

RANKS = 2



# ------------------------------------------------------------------------------
#
class MyMaster(rp.raptor.Master):
    '''
    This class provides the communication setup for the task overlay: it will
    set up the request / response communication queues and provide the endpoint
    information to be forwarded to the workers.
    '''

    # --------------------------------------------------------------------------
    #
    def __init__(self, cfg):

        self._cnt = 0
        self._submitted = defaultdict(int)
        self._collected = defaultdict(int)

        # initialize the task overlay base class.  That base class will ensure
        # proper communication channels to the pilot agent.
        super().__init__(cfg=cfg)

        self._sleep = self._cfg.sleep


    # --------------------------------------------------------------------------
    #
    def submit(self):

       # self._prof.prof('create_start')

        # create additional tasks to be distributed to the workers.

        #tds = list()

        #self._prof.prof('create_stop')

        # wait for outstanding tasks to complete
        while True:

            completed = sum(self._collected.values())
            submitted = sum(self._submitted.values())

            if submitted:
                # request_cb has been called, so we can wait for completion

                self._log.info('=== submit done?: %d >= %d ', completed, submitted)

                if completed >= submitted:
                    break

            time.sleep(1)

        self._log.info('=== submit done!')


    # --------------------------------------------------------------------------
    #
    def request_cb(self, tasks):

        for task in tasks:

            self._log.debug('=== request_cb %s\n', task['uid'])

            mode = task['description']['mode']
            uid  = task['description']['uid']

            self._submitted[mode] += 1

            # for each `function` mode task, submit one more `proc` mode request
            if mode == rp.TASK_FUNC:
                self.submit_tasks(rp.TaskDescription(
                    {'uid'       : 'extra' + uid,
                   # 'timeout'   : 10,
                     'mode'      : rp.TASK_PROC,
                     'ranks'     : RANKS,
                     'executable': '/bin/sh',
                     'arguments' : ['-c',
                                    'sleep %d; ' % self._sleep +
                                    'echo "hello $RP_RANK/$RP_RANKS: '
                                          '$RP_TASK_ID"'],
                     'raptor_id' : 'master.000000'}))

        return tasks


    # --------------------------------------------------------------------------
    #
    def result_cb(self, tasks):
        '''
        Log results.

        Log file is named by the master tasks UID.
        '''
        for task in tasks:

            mode = task['description']['mode']
            self._collected[mode] += 1

            # NOTE: `state` will be `AGENT_EXECUTING`
            self._log.info('=== result_cb  %s: %s [%s] [%s]',
                            task['uid'],
                            task['state'],
                            task['stdout'],
                            task['return_value'])

            # Note that stdout is part of the master task result.
            print('id: %s [%s]:\n    out: %s\n    ret: %s\n'
                 % (task['uid'], task['state'], task['stdout'],
                    task['return_value']))


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    # This master script runs as a task within a pilot allocation.  The purpose
    # of this master is to (a) spawn a set or workers within the same
    # allocation, (b) to distribute work items (`hello` function calls) to those
    # workers, and (c) to collect the responses again.
    cfg_fname    = str(sys.argv[1])
    cfg          = ru.Config(cfg=ru.read_json(cfg_fname))
    cfg.rank     = int(sys.argv[2])

    n_workers        = cfg.n_workers
    nodes_per_worker = cfg.nodes_per_worker
    cores_per_node   = cfg.cores_per_node
    gpus_per_node    = cfg.gpus_per_node
    descr            = cfg.worker_descr
    pwd              = os.getcwd()

    # one node is used by master.  Alternatively (and probably better), we could
    # reduce one of the worker sizes by one core.  But it somewhat depends on
    # the worker type and application workload to judge if that makes sense, so
    # we leave it for now.

    # create a master class instance - this will establish communication to the
    # pilot agent
    master = MyMaster(cfg)

    # insert `n` worker tasks into the agent.  The agent will schedule (place)
    # those workers and execute them.  Insert one smaller worker (see above)
    # NOTE: this assumes a certain worker size / layout
    out('workers: %d' % n_workers)
    descr['ranks']         = nodes_per_worker * cores_per_node
    descr['gpus_per_rank'] = nodes_per_worker * gpus_per_node
    worker_ids = master.submit_workers(
                 [rp.TaskDescription(descr) for _ in range(n_workers)])

    # wait until `m` of those workers are up
    # This is optional, work requests can be submitted before and will wait in
    # a work queue.
    # FIXME
    master.wait_workers(count=1)

    out('start')
    master.start()
    out('submit')
    master.submit()

    # let some time pass for client side tasks to complete
    time.sleep(600)

    out('stop')
    # TODO: can be run from thread?
    master.stop()
    out('join')

    # TODO: worker state callback
    master.join()
    out('done')

    # TODO: expose RPC hooks


# ------------------------------------------------------------------------------
