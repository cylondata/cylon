#!/usr/bin/env python3

'''
Demonstrate the "raptor" features for remote Task management.

This script and its supporting files may use relative file paths. Run from the
directory in which you found it.

Refer to the ``raptor.cfg`` file in the same directory for configurable run time
details.

By default, this example uses the ``local.localhost`` resource with the
``local`` access scheme where RP oversubscribes resources to emulate multiple
nodes.

In this example, we
  - Launch one or more raptor "master" task(s), which self-submits additional
    tasks (results are logged in the master's `result_cb` callback).
  - Stage scripts to be used by a raptor "Worker"
  - Provision a Python virtual environment with
    :py:func:`~radical.pilot.prepare_env`
  - Submit several tasks that will be routed through the master(s) to the
    worker(s).
  - Submit a non-raptor task in the same Pilot environment

'''

import os
import sys
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import readfile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console

import radical.utils as ru
import radical.pilot as rp


# To enable logging, some environment variables need to be set.
# Ref
# * https://radicalpilot.readthedocs.io/en/stable/overview.html#what-about-logging
# * https://radicalpilot.readthedocs.io/en/stable/developer.html#debugging
# For terminal output, set RADICAL_LOG_TGT=stderr or RADICAL_LOG_TGT=stdout
logger = ru.Logger('raptor')
PWD    = os.path.abspath(os.path.dirname(__file__))
RANKS  = 72


# ------------------------------------------------------------------------------
#
@rp.pythontask
def cylon_join(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':5000, 'it': 10}):

    import time
    import argparse

    import pandas as pd

    from numpy.random import default_rng
    from pycylon.frame import CylonEnv, DataFrame
    from pycylon.net import MPIConfig
    
    comm = comm
    data = data

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))
    
    
    for i in range(data['it']):
        env.barrier()
        t1 = time.time()
        df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)

    if env.rank == 0:
        pass
    env.finalize()
    
    
@rp.pythontask
def cylon_slice(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':5000, 'it': 10}):

    import time
    import argparse

    import pandas as pd

    from numpy.random import default_rng
    from pycylon.frame import CylonEnv, DataFrame
    from pycylon.net import MPIConfig
    
    comm = comm
    data = data

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    #data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    #df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))
    
    
    for i in range(data['it']):
        env.barrier()
        t1 = time.time()
        df3 = df1[max_val * 0.2 :max_val * u, env] # distributed slice
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)

    if env.rank == 0:
        pass
    env.finalize()

@rp.pythontask
def cylon_groupby(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':5000, 'it': 10}):

    import time
    import argparse

    import pandas as pd

    from numpy.random import default_rng
    from pycylon.frame import CylonEnv, DataFrame
    from pycylon.net import MPIConfig
    
    comm = comm
    data = data

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    
    
    for i in range(data['it']):
        env.barrier()
        t1 = time.time()
        
        df3 = df1.groupby(by=0).agg({
            "1": "sum",
            "2": "min"
        })

        df4 = df1.groupby(by=0).min()

        df5 = df1.groupby(by=[0, 1]).max()
        
        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)

    if env.rank == 0:
        pass
    env.finalize()
    
@rp.pythontask
def cylon_sort(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':5000, 'it': 10}):

    import time
    import argparse

    import pandas as pd

    from numpy.random import default_rng
    from pycylon.frame import CylonEnv, DataFrame
    from pycylon.net import MPIConfig
    
    comm = comm
    data = data

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    #data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    #df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))
    
    
    for i in range(data['it']):
        env.barrier()
        t1 = time.time()
        #df3 = df1.merge(df2, on=[0], algorithm='sort', env=env)
        #print("Distributed Sort with sort options", env.rank)
        bins = env.world_size * 2
        #df3 = df1.sort_values(by=[0], num_bins=bins, num_samples=bins, env=env)
        df3 = df1.sort_values(by=[0], env=env)
        #print(df3)

        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df3))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)

    if env.rank == 0:
        pass
    env.finalize()
    

@rp.pythontask
def cylon_concat(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':5000, 'it': 10}):

    import time
    import argparse

    import pandas as pd
    import pycylon as cn
    from numpy.random import default_rng
    from pycylon.frame import CylonEnv, DataFrame
    from pycylon.net import MPIConfig
    
    comm = comm
    data = data

    config = MPIConfig(comm)
    env = CylonEnv(config=config, distributed=True)

    u = data['unique']

    if data['scaling'] == 'w':  # weak
        num_rows = data['rows']
        max_val = num_rows * env.world_size
    else:  # 's' strong
        max_val = data['rows']
        num_rows = int(data['rows'] / env.world_size)

    rng = default_rng(seed=env.rank)
    data1 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data2 = rng.integers(0, int(max_val * u), size=(num_rows, 2))
    data3 = rng.integers(0, int(max_val * u), size=(num_rows, 2))

    df1 = DataFrame(pd.DataFrame(data1).add_prefix("col"))
    df2 = DataFrame(pd.DataFrame(data2).add_prefix("col"))
    df3 = DataFrame(pd.DataFrame(data3).add_prefix("col"))
    
    
    for i in range(data['it']):
        env.barrier()
        t1 = time.time()
        df4 = cn.concat(axis=0, objs=[df1, df2, df3], env=env)
        #print(df4)

        env.barrier()
        t2 = time.time()
        t = (t2 - t1)
        sum_t = comm.reduce(t)
        tot_l = comm.reduce(len(df4))

        if env.rank == 0:
            avg_t = sum_t / env.world_size
            print("### ", data['scaling'], env.world_size, num_rows, max_val, i, avg_t, tot_l)

    if env.rank == 0:
        pass
    env.finalize()
    
    

# ------------------------------------------------------------------------------
#
def task_state_cb(task, state):
    logger.info('task %s: %s', task.uid, state)
    if state == rp.FAILED:
        logger.error('task %s failed', task.uid)


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    import time

    if len(sys.argv) < 2:
        cfg_file = '%s/raptor.cfg' % PWD

    else:
        cfg_file = sys.argv[1]

    cfg              = ru.Config(cfg=ru.read_json(cfg_file))

    cores_per_node   = cfg.cores_per_node
    gpus_per_node    = cfg.gpus_per_node
    n_masters        = cfg.n_masters
    n_workers        = cfg.n_workers
    masters_per_node = cfg.masters_per_node
    nodes_per_worker = cfg.nodes_per_worker
    n_rows           = cfg.no_of_rows

    # we use a reporter class for nicer output
    report = ru.Reporter(name='radical.pilot')
    report.title('Raptor example (RP version %s)' % rp.version)

    session = rp.Session()
    try:
        pd = rp.PilotDescription(cfg.pilot_descr)

        #pd.cores  = cores_per_node + 2
        pd.nodes = ((cores_per_node + 2)/42)
        pd.gpus   = 0
        pd.runtime = 60

        pmgr = rp.PilotManager(session=session)
        tmgr = rp.TaskManager(session=session)
        tmgr.register_callback(task_state_cb)

        pilot = pmgr.submit_pilots(pd)
        tmgr.add_pilots(pilot)

        pmgr.wait_pilots(uids=pilot.uid, state=[rp.PMGR_ACTIVE])

        report.info('Stage files for the worker `my_hello` command.\n')
        # See raptor_worker.py.
        pilot.stage_in({'source': ru.which('radical-pilot-hello.sh'),
                        'target': 'radical-pilot-hello.sh',
                        'action': rp.TRANSFER})

        # Issue an RPC to provision a Python virtual environment for the later
        # raptor tasks.  Note that we are telling prepare_env to install
        # radical.pilot and radical.utils from sdist archives on the local
        # filesystem. This only works for the default resource, local.localhost.
        report.info('Call pilot.prepare_env()... ')
        pilot.prepare_env(env_name='CYLON',
                          env_spec={'type' : 'venv',
                                    'path' : '$HOME/CYLON',
                                    'setup': []})
        report.info('done\n')

        # Launch a raptor master task, which will launch workers and self-submit
        # some additional tasks for illustration purposes.

        master_ids = [ru.generate_id('master.%(item_counter)06d',
                                     ru.ID_CUSTOM, ns=session.uid)
                      for _ in range(n_masters)]

        tds = list()
        #f = open("output-{cores_per_node}-{n_rows}.log", "w")
        for i in range(n_masters):
            td = rp.TaskDescription(cfg.master_descr)
            td.mode           = rp.RAPTOR_MASTER
            td.uid            = master_ids[i]
            td.arguments      = [cfg_file, i]
            td.cpu_processes  = 1
            td.cpu_threads    = 1
            td.named_env      = 'rp'
            td.input_staging  = [{'source': '%s/raptor_master.py' % PWD,
                                  'target': 'raptor_master.py',
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS},
                                 {'source': '%s/raptor_worker.py' % PWD,
                                  'target': 'raptor_worker.py',
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS},
                                 {'source': cfg_file,
                                  'target': os.path.basename(cfg_file),
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS}
                                ]
            tds.append(td)

        if len(tds) > 0:
            report.info('Submit raptor master(s) %s\n'
                       % str([t.uid for t in tds]))
            task  = tmgr.submit_tasks(tds)
            if not isinstance(task, list):
                task = [task]

            states = tmgr.wait_tasks(
                uids=[t.uid for t in task],
                state=rp.FINAL + [rp.AGENT_EXECUTING],
                timeout=3600
            )
            logger.info('Master states: %s', str(states))

        tds = list()
        bson = list()
        bson.append(cylon_sort(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':n_rows, 'it': 10}))
        bson.append(cylon_join(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':n_rows, 'it': 10}))
        for i in range(2):

            #bson = cylon_join(comm=None, data={'unique': 0.9, 'scaling': 's', 'rows':n_rows, 'it': 10}) # For join test
            #bson = cylon_groupby(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':n_rows, 'it': 10})
            #bson = cylon_sort(comm=None, data={'unique': 0.9, 'scaling': 's', 'rows':n_rows, 'it': 10})
            #bson = cylon_slice(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':n_rows, 'it': 10})
            #bson = cylon_concat(comm=None, data={'unique': 0.9, 'scaling': 'w', 'rows':n_rows, 'it': 10})
            
            tds.append(rp.TaskDescription({
                'uid'             : 'task.cylon.w.%06d' % i,
                'mode'            : rp.TASK_FUNC,
                'ranks'           : cores_per_node,
                'function'        : bson[i],
                'raptor_id'       : master_ids[i % n_masters]}))


        if len(tds) > 0:
            t3 = time.time()
            report.info('Submit tasks %s.\n' % str([t.uid for t in tds]))
            tasks = tmgr.submit_tasks(tds)

            logger.info('Wait for tasks %s', [t.uid for t in tds])
            tmgr.wait_tasks(uids=[t.uid for t in tasks])

            for task in tasks:
                report.info('id: %s [%s]:\n    out: %s\n    ret: %s\n'
                      % (task.uid, task.state, task.stdout, task.return_value))
            t4 = time.time()
            t5 = (t4-t3)
            print("### Heterogeneous Execution time: ", t5)

    finally:
        session.close(download=True)

    report.info('Logs from the master task should now be in local files \n')
    report.info('like %s/%s/%s.log\n' % (session.uid, pilot.uid, master_ids[0]))

# ------------------------------------------------------------------------------
