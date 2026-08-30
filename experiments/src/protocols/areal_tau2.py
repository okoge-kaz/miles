"""Identity contract for AReaL Tau2 conversational environment rollouts."""

AREAL_TAU2_DATASET = "inclusionAI/AReaL-tau2-data"
AREAL_TAU2_REVISION = "1322eae337f836fe0e19bae14dab1eefc26bc983"
AREAL_TAU2_VERIFIER = "areal_tau2_environment"
AREAL_TAU2_INTERACTION_MODE = "stateful_multi_turn_user_simulator_environment"
AREAL_TAU2_POLICY = "areal-tau2-user-simulator-terminal-state-rl"
AREAL_TAU2_DOMAINS = ("airline", "retail", "telecom")
AREAL_TAU2_EXPECTED_COUNTS = {
    "airline": 1_148,
    "retail": 563,
    "telecom": 271,
}
AREAL_TAU2_EXPECTED_ROWS = sum(AREAL_TAU2_EXPECTED_COUNTS.values())
AREAL_TAU2_RAW_SHA256 = "5af43e8c58d58b4b3eb38cdd4650a3e93c89fa6d25382c2ef9981d8420d90f41"
AREAL_TAU2_DB_SHA256 = {
    "tau2_rl_database/tau2_airline_db.json": (
        "1af9fea6e03ca7ca15a22bb3fcaf3e351393e3fc9070b6777947da8996f7531b"
    ),
    "tau2_rl_database/tau2_airline_new_db_1.json": (
        "b69dbaee93456de6a612bd2ea5174225afe746be4e109c075a9f4db15aaa56c3"
    ),
    "tau2_rl_database/tau2_airline_new_db_2.json": (
        "a8a5201ee4f4959bd38787089957029846f79760bf26a9756f996aab166cf1ad"
    ),
    "tau2_rl_database/tau2_airline_new_db_3.json": (
        "82255bf2b6c92de4328b401c1a7d8b844d4bdd15746fe00575ee36b2044732b1"
    ),
    "tau2_rl_database/tau2_retail_new_db_1.json": (
        "785b3fe40f8f9e9ac033bf3071c3f3bd27da4108ce68532fb4170be5d3a1c07e"
    ),
    "tau2_rl_database/tau2_retail_new_db_2.json": (
        "4cfa2f845239052f0e062328dbfb2b41477639079c123e957a473f9e192c0e4a"
    ),
    "tau2_rl_database/tau2_retail_new_db_3.json": (
        "2d872f166fa11da78df4dba12905e8dcf2e71d1873ec2a8f0c26c7d2a686e6e7"
    ),
    "tau2_rl_database/tau2_retail_new_db_4.json": (
        "00b2095d6351c958b546561f791663c413e83346b1becf59cd55a1c3a49b749b"
    ),
    "tau2_rl_database/tau2_telecom_db.toml": (
        "562d647ef9d7df8df91eafd8ee76036e707c8f9e32aaedc4a7fde06975aea2c0"
    ),
}

__all__ = [
    "AREAL_TAU2_DATASET",
    "AREAL_TAU2_DB_SHA256",
    "AREAL_TAU2_DOMAINS",
    "AREAL_TAU2_EXPECTED_COUNTS",
    "AREAL_TAU2_EXPECTED_ROWS",
    "AREAL_TAU2_INTERACTION_MODE",
    "AREAL_TAU2_POLICY",
    "AREAL_TAU2_RAW_SHA256",
    "AREAL_TAU2_REVISION",
    "AREAL_TAU2_VERIFIER",
]
