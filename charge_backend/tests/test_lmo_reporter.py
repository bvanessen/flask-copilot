###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

import asyncio
import json

from charge_backend.backend_helper_funcs import FlaskRunSettings
from charge_backend.flask_experiment import GraphContext
from charge_backend.lmo import lmo_reporter


class FakeWebSocket:
    """Records send_json calls so tests can inspect what was transmitted."""

    def __init__(self):
        self.messages = []

    async def send_json(self, data):
        self.messages.append(data)


def _mol(smiles="CCO", density=1.1, sascore=2.0, bandgap=0.3):
    return {
        "smiles": smiles,
        "density": density,
        "sascore": sascore,
        "bandgap": bandgap,
    }


def test_format_hover_contains_properties():
    hover = lmo_reporter.format_hover(_mol())
    assert "`CCO`" in hover
    assert "1.100" in hover  # density, 3 decimals
    assert "2.000" in hover  # sascore
    assert "0.30" in hover  # bandgap, 2 decimals


def test_add_root_node_streams_node():
    g = GraphContext()
    ws = FakeWebSocket()
    node = asyncio.run(lmo_reporter.add_root_node(g, ws, _mol(), FlaskRunSettings()))

    assert node.id in g.node_ids
    assert node.level == 0
    assert node.parentId is None
    assert node.density == 1.1
    node_msgs = [m for m in ws.messages if m["type"] == "node"]
    assert len(node_msgs) == 1
    assert node_msgs[0]["node"]["smiles"] == "CCO"


def test_add_candidate_node_creates_real_edge():
    g = GraphContext()
    ws = FakeWebSocket()
    root = asyncio.run(lmo_reporter.add_root_node(g, ws, _mol(), FlaskRunSettings()))
    child = asyncio.run(
        lmo_reporter.add_candidate_node(
            g,
            ws,
            _mol(smiles="CCN"),
            level=1,
            parent_node_id=root.id,
            is_best=True,
            run_settings=FlaskRunSettings(),
        )
    )

    # Parent relationship recorded and a real edge created by GraphContext
    assert g.parents[child.id] == root.id
    assert any(e.fromNode == root.id and e.toNode == child.id for e in g.edges.values())
    assert child.highlight == "yellow"

    edge_msgs = [m for m in ws.messages if m["type"] == "edge"]
    assert len(edge_msgs) == 1
    assert edge_msgs[0]["edge"]["fromNode"] == root.id
    assert edge_msgs[0]["edge"]["toNode"] == child.id


def test_add_candidate_node_not_best_is_normal():
    g = GraphContext()
    ws = FakeWebSocket()
    root = asyncio.run(lmo_reporter.add_root_node(g, ws, _mol(), FlaskRunSettings()))
    child = asyncio.run(
        lmo_reporter.add_candidate_node(
            g,
            ws,
            _mol(smiles="CCN"),
            level=1,
            parent_node_id=root.id,
            is_best=False,
            run_settings=FlaskRunSettings(),
        )
    )
    assert child.highlight == "normal"


def test_send_final_summary_sends_complete():
    ws = FakeWebSocket()
    asyncio.run(lmo_reporter.send_final_summary(ws, "CCO", "density", 1.234))

    assert ws.messages[-1] == {"type": "complete"}
    response = ws.messages[-2]
    assert response["type"] == "response"
    assert "CCO" in response["message"]["message"]
    assert "density=1.234" in response["message"]["message"]
    assert response["message"]["smiles"] == "CCO"


def test_send_iteration_messages():
    ws = FakeWebSocket()
    asyncio.run(lmo_reporter.send_iteration_start(ws, 0, 3))
    asyncio.run(lmo_reporter.send_iteration_progress(ws, 0, 3, "CCO", "density", 1.5))

    assert ws.messages[0]["message"]["message"] == "Iteration 1/3"
    progress = ws.messages[1]["message"]["message"]
    assert "Iteration 1/3 complete" in progress
    assert "density=1.500" in progress


def test_send_status_edge_computing_then_complete():
    ws = FakeWebSocket()
    asyncio.run(lmo_reporter.send_status_edge(ws, "edge_x", "node_0", "node_1", False))
    asyncio.run(lmo_reporter.send_status_edge(ws, "edge_x", "node_0", "node_1", True))

    assert ws.messages[0]["type"] == "edge"
    assert ws.messages[0]["edge"]["status"] == "computing"
    assert ws.messages[0]["edge"]["label"] == "Optimizing"
    assert ws.messages[1]["type"] == "edge_update"
    assert ws.messages[1]["edge"]["status"] == "complete"
    assert ws.messages[1]["edge"]["label"] == "Completed"


def test_persist_molecules_writes_full_records(tmp_path):
    path = tmp_path / "mols.json"
    records = [_mol(), _mol(smiles="CCN")]
    lmo_reporter.persist_molecules(records, str(path))

    saved = json.loads(path.read_text())
    assert saved == records
