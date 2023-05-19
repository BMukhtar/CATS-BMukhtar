#!/usr/bin/python

import asyncio
import time
import random

async def task1():
    print("task 1 started")
    wait = random.randint(0, 3)
    await asyncio.sleep(wait)
    print("task 1 finished")


async def task2():
    print("task 2 started")
    wait = random.randint(0, 3)
    await asyncio.sleep(wait)
    print("task 2 finished")


async def task3():
    print("task 3 started")
    wait = random.randint(0, 3)
    await asyncio.sleep(wait)
    print("task 3 finished")


async def main():
    for x in range(2):
        t1 = task1()
        t2 = task2()
        t3 = task3()
        await asyncio.gather(t1,t2,t3)
        time.sleep(1)
        print('----------------------------')

t1 = time.perf_counter()
asyncio.run(main())
t2 = time.perf_counter()

print(f'Total time elapsed: {t2-t1:0.2f} seconds')
#%%
