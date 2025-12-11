import json
import os
import time

import polars as pl
import polars_cloud as pc

from queries.polars.q1 import q as q1
from queries.polars.q2 import q as q2
from queries.polars.q3 import q as q3
from queries.polars.q4 import q as q4
from queries.polars.q5 import q as q5
from queries.polars.q6 import q as q6
from queries.polars.q7 import q as q7
from queries.polars.q8 import q as q8
from queries.polars.q9 import q as q9
from queries.polars.q10 import q as q10
from queries.polars.q11 import q as q11
from queries.polars.q12 import q as q12
from queries.polars.q13 import q as q13
from queries.polars.q14 import q as q14
from queries.polars.q15 import q as q15
from queries.polars.q16 import q as q16
from queries.polars.q17 import q as q17
from queries.polars.q18 import q as q18
from queries.polars.q19 import q as q19
from queries.polars.q20 import q as q20
from queries.polars.q21 import q as q21
from queries.polars.q22 import q as q22

PDSH_BUCKET = os.path.normpath(os.environ.get("PDSH_BUCKET", "polars-pdsh/scale-1000.0/200"))
STORAGE_OPTIONS = json.loads(os.environ.get("STORAGE_OPTIONS", "{}"))
PC_CLUSTER_CONTEXT = os.environ.get("PC_CLUSTER_CONTEXT", 0)


def _scan_ds(table_name: str) -> pl.LazyFrame:
    path = f"s3://{PDSH_BUCKET}/{table_name}/"
    return pl.scan_parquet(path, storage_options=STORAGE_OPTIONS)


lineitem = _scan_ds("lineitem")
orders = _scan_ds("orders")
customer = _scan_ds("customer")
region = _scan_ds("region")
nation = _scan_ds("nation")
supplier = _scan_ds("supplier")
part = _scan_ds("part")
part_supp = _scan_ds("partsupp")

kwargs = {
    "lineitem": lineitem,
    "orders": orders,
    "customer": customer,
    "region": region,
    "nation": nation,
    "supplier": supplier,
    "part": part,
    "partsupp": part_supp,
}

queries = [
    q1(**kwargs),
    q2(**kwargs),
    q3(**kwargs),
    q4(**kwargs),
    q5(**kwargs),
    q6(**kwargs),
    q7(**kwargs),
    q8(**kwargs),
    q9(**kwargs),
    q10(**kwargs),
    q11(**kwargs),
    q12(**kwargs),
    q13(**kwargs),
    q14(**kwargs),
    q15(**kwargs),
    q16(**kwargs),
    q17(**kwargs),
    q18(**kwargs),
    q19(**kwargs),
    q20(**kwargs),
    q21(**kwargs),
    q22(**kwargs),
]


class PatchedContext(pc.ClusterContext):
    def start(self, *, wait: bool = False) -> None:
        pass

    def stop(self, *, wait: bool = False) -> None:
        pass

ctx: PatchedContext | pc.ComputeContext
if PC_CLUSTER_CONTEXT:
    ctx = PatchedContext(
        compute_address=os.environ.get("PC_SCHEDULER_ADDRESS", "localhost"),
        compute_port=int(os.environ.get("PC_SCHEDULER_PORT", 5051)),
        insecure=bool(os.environ.get("PC_SCHEDULER_INSECURE", "1") == "1")
    )
else:
    pc.authenticate()

    ctx = pc.ComputeContext(
        workspace="polars-ritchie-dev",
        instance_type="m6i.xlarge",
        cluster_size=32,
        storage=128,
    )

ctx.start(wait=True)

print("started cluster")


timings: list[float | None] = []
for i, q in enumerate(queries):
    i += 1
    print(f"run q{i}")
    start_time = time.time()
    try:
        result = (
            q.remote(ctx) # type: ignore[arg-type]
            .distributed(shuffle_compression="zstd")
            .execute()
            .await_result()
        )
        print(result.head)
        execution_time = time.time() - start_time
        print(f"q{i} executed in: {execution_time:.2f} seconds")
        timings.append(execution_time)
    except pl.exceptions.ComputeError as e:
        print("remote query failed:", e)
        timings.append(None)

print("timings: ", timings)

ctx.stop(wait=True)
