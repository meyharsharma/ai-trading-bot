# ERC-8004 Deployment Notes

> Day-1 deliverable for the on-chain layer. Locks the chain we target and the
> contract addresses our Python clients call into.

## TL;DR

We target **Base Sepolia (chain id `84532`)** for the hackathon submission.

| Contract | Address |
|---|---|
| `IdentityRegistry`   | `0x8004A818BFB912233c491871b3d84c89A494BD9e` |
| `ReputationRegistry` | `0x8004B663056A597Dffe9eCcC1965A193B7388713` |
| `ValidationRegistry` | _TBD — pull from `erc-8004/erc-8004-contracts` deployments json on Day 2_ |

The ERC-8004 reference deployments use the **same addresses across every
EVM testnet** (Ethereum Sepolia, Base Sepolia, Linea Sepolia, Hedera Testnet),
which means a fallback chain swap is a one-line change in `.env`.

## Why Base Sepolia

1. **Cheapest gas of the supported testnets.** Validation artifacts are
   submitted *every loop* (~288/day) — fees matter. Base Sepolia gas is
   sub-cent and faucets are easy.
2. **Fastest blocks (~2s).** Our 5-minute trading loop wants near-instant
   confirmations so the artifact lands in the same cycle as the trade.
3. **Hackathon Capital Sandbox vault is most likely deployed here too.** Base
   is the canonical EVM hackathon chain in 2026 and the brief frames the vault
   as "where the agent operates" — co-locating identity + vault on Base avoids
   any cross-chain bridging in the demo.
4. **Reference deployment exists today.** The `erc-8004/erc-8004-contracts`
   repo lists Base Sepolia under the canonical address set (verified
   2026-04-06).

Mainnet ERC-8004 went live 2026-01-29 on Ethereum L1, but we deliberately stay
on testnet — the brief is unambiguous that the on-chain track is judged on
**validation quality**, not on whether you burned real ETH.

## What we actually call

### Identity (one-shot, at agent boot)

```solidity
function register(string agentURI) external returns (uint256 agentId)
```

`agentURI` points to an off-chain JSON manifest (model name, strategy
description, owner, repo URL). We store the resulting `agentId` in
`config/agent_id.json` so the trading loop never re-registers.

### Validation Artifact (every decision, atomically with the trade)

```solidity
function validationRequest(
    address validatorAddress,   // self for the hackathon — we are our own validator
    uint256 agentId,
    string  requestURI,         // "ipfs://..." or "data:application/json;base64,..."
    bytes32 requestHash         // canonical_hash(ValidationArtifact)
) external
```

The `requestHash` is the sha256 of our `ValidationArtifact` pydantic model
(see `src/agent/state/models.py::canonical_hash`). This is the **single source
of truth** linking an off-chain decision to its on-chain fingerprint.

Optionally we deploy our own `AgentArtifacts.sol` (a 30-line wrapper that
emits an indexed `ArtifactSubmitted` event) so the demo video can show a
filterable on-chain history without writing a subgraph. This is purely a
nice-to-have on top of the standard `ValidationRegistry` write.

### Reputation (post-close, optional)

```solidity
function giveFeedback(uint256 agentId, int128 value, uint8 valueDecimals,
                     string tag1, string tag2,
                     string endpoint, string feedbackURI, bytes32 feedbackHash)
```

We use this *after* a position closes to anchor realized PnL as objective
on-chain reputation. Only attempted in Day 5 if time permits.

## Hackathon Capital Sandbox vault

The brief mandates that the agent "operate through the Hackathon Capital
Sandbox vault and risk router" but does **not** publish an address. Day-1
status:

- **Address:** _unknown_ — pending organizer DM / Discord post.
- **Abstraction:** `src/agent/chain/vault.py` ships with a configurable
  `VAULT_ADDRESS` env var and a documented `route_intent()` interface so we
  can drop in the real address the moment we get it without touching the
  trading loop.
- **Fallback:** if no vault address is published by Day 4, our `vault.py`
  emits a self-contained "vault intent" event via `AgentArtifacts.sol` so the
  validation chain remains unbroken.

## Wallet & gas

- **Network RPC:** `https://sepolia.base.org` (env: `RPC_URL`).
- **Wallet:** generated via `eth-account`, key in `.env` as `PRIVATE_KEY`
  (never committed). Address pinned in `config/agent_id.json` after
  registration.
- **Faucet:** Coinbase Base Sepolia faucet. Top up to ~0.05 ETH; one
  validation artifact write costs <100k gas.

## Sources

- [EIP-8004 spec](https://eips.ethereum.org/EIPS/eip-8004)
- [erc-8004/erc-8004-contracts](https://github.com/erc-8004/erc-8004-contracts)
- [ChaosChain reference impl](https://github.com/ChaosChain/trustless-agents-erc-ri)
- [8004scan.io](https://8004scan.io) — explorer for deployed agents

_Last verified: 2026-04-06._
