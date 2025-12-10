// SPDX-License-Identifier: MIT
pragma solidity 0.8.24;

/* ---------- Subtensor precompiles ---------- */

address constant INEURON_ADDR   = 0x0000000000000000000000000000000000000804;
address constant ISTAKING_ADDR  = 0x0000000000000000000000000000000000000805;

interface INeuron {
    function burnedRegister(uint16 netuid, bytes32 hotkey) external payable;
}

interface IStaking {
    function transferStake(
        bytes32 destination_coldkey,
        bytes32 hotkey,
        uint256 origin_netuid,
        uint256 destination_netuid,
        uint256 amount
    ) external;

    function getStake(
        bytes32 hotkey,
        bytes32 coldkey,
        uint256 netuid
    ) external view returns (uint256);

    function getTotalHotkeyStake(bytes32 hotkey) external view returns (uint256);
}

/* ---------- Contract ---------- */

contract UltraSimplifiedMultiTrusteeDistributor {
    address public immutable trustee1;
    address public immutable trustee2;
    address public immutable trustee3;
    uint256 public constant REQUIRED_VOTES = 2;

    address public immutable deployer;
    bool public paused;
    bool public burned;

    uint16  public subnet;
    bytes32 public contractHotkey;
    bytes32 public contractColdkey;

    uint256 public latestScore;

    bytes32 public pendingRecipient;
    uint256 public pendingScore;
    uint256 public voteCount;
    mapping(address => bool) public hasVoted;

    address public lastAttemptedBy;
    uint256 public lastAttemptTime;
    bytes32 public lastAttemptedRecipient;
    uint256 public totalVoteAttempts;

    event VoteCast(address indexed trustee, bytes32 recipientColdkey, uint256 newScore);
    event AllRewardsDistributed(bytes32 recipientColdkey, uint256 stakeAmount, uint256 newScore);
    event VoteReset();
    event ContractPaused(bool isPaused);
    event ContractBurned();
    event HotkeyRegistered(bytes32 hotkey, uint16 netuid);
    event ContractColdkeySet(bytes32 coldkey);
    event DebugVoteProgress(uint256 currentVotes, uint256 required);

    modifier notPaused() {
        require(!paused, "paused");
        _;
    }

    modifier notBurned() {
        require(!burned, "burned");
        _;
    }

    modifier onlyTrustee() {
        require(isTrustee(msg.sender), "not trustee");
        _;
    }

    modifier onlyDeployer() {
        require(msg.sender == deployer, "not deployer");
        _;
    }

    constructor(address _trustee1, address _trustee2, address _trustee3) payable {
        require(_trustee1 != address(0), "invalid trustee1");
        require(_trustee2 != address(0), "invalid trustee2");
        require(_trustee3 != address(0), "invalid trustee3");

        trustee1 = _trustee1;
        trustee2 = _trustee2;
        trustee3 = _trustee3;

        deployer = msg.sender;
    }

    receive() external payable {}

    function registerHotkey(uint16 netuid, bytes32 hotkey)
        external
        payable
        onlyDeployer
        notBurned
    {
        require(contractHotkey == bytes32(0), "hotkey already set");
        bytes memory data = abi.encodeWithSelector(
            INeuron.burnedRegister.selector,
            netuid,
            hotkey
        );
        (bool success, ) = INEURON_ADDR.call{gas: gasleft(), value: msg.value}(data);
        subnet = netuid;
        contractHotkey = hotkey;
        emit HotkeyRegistered(hotkey, netuid);
    }

    function setContractColdkey(bytes32 coldkey) external onlyDeployer notBurned {
        require(contractColdkey == bytes32(0), "coldkey already set");
        require(coldkey != bytes32(0), "invalid coldkey");
        contractColdkey = coldkey;
        emit ContractColdkeySet(coldkey);
    }

    function releaseReward(bytes32 recipientColdkey, uint256 newScore)
        external
        onlyTrustee
        notPaused
        notBurned
    {
        require(recipientColdkey != bytes32(0), "invalid recipient");
        require(contractHotkey != bytes32(0), "hotkey not set");
        require(subnet != 0, "subnet not set");
        require(contractColdkey != bytes32(0), "coldkey not set");

        lastAttemptedBy = msg.sender;
        lastAttemptTime = block.timestamp;
        lastAttemptedRecipient = recipientColdkey;
        totalVoteAttempts += 1;

        if (voteCount > 0 && (pendingRecipient != recipientColdkey || pendingScore != newScore)) {
            _resetVotes();
        }

        require(!hasVoted[msg.sender], "already voted");

        if (voteCount == 0) {
            pendingRecipient = recipientColdkey;
            pendingScore = newScore;
        }

        hasVoted[msg.sender] = true;
        voteCount++;

        emit VoteCast(msg.sender, recipientColdkey, newScore);
        emit DebugVoteProgress(voteCount, REQUIRED_VOTES);

        if (voteCount >= REQUIRED_VOTES) {
            uint256 stakeOwned =
                IStaking(ISTAKING_ADDR).getStake(contractHotkey, contractColdkey, uint256(subnet));
            require(stakeOwned > 0, "no stake");

            latestScore = newScore;
            _resetVotes();

            bytes memory data = abi.encodeWithSelector(
                IStaking.transferStake.selector,
                recipientColdkey,
                contractHotkey,
                uint256(subnet),
                uint256(subnet),
                stakeOwned
            );
            (bool success, ) = ISTAKING_ADDR.call{gas: gasleft()}(data);

            emit AllRewardsDistributed(recipientColdkey, stakeOwned, newScore);
        }
    }

    function setPaused(bool _paused) external onlyDeployer {
        paused = _paused;
        emit ContractPaused(_paused);
    }

    function burn() external onlyDeployer {
        burned = true;
        emit ContractBurned();
    }

    function isTrustee(address addr) public view returns (bool) {
        return (addr == trustee1 || addr == trustee2 || addr == trustee3);
    }

    function getBalance() external view returns (uint256) {
        return address(this).balance;
    }

    function getOwnedStake() public view returns (uint256) {
        if (contractHotkey == bytes32(0) || contractColdkey == bytes32(0) || subnet == 0) {
            return 0;
        }
        return IStaking(ISTAKING_ADDR).getStake(contractHotkey, contractColdkey, uint256(subnet));
    }

    function getTotalHotkeyStake() public view returns (uint256) {
        if (contractHotkey == bytes32(0)) return 0;
        return IStaking(ISTAKING_ADDR).getTotalHotkeyStake(contractHotkey);
    }

    function getVotingStatus()
        external
        view
        returns (bytes32 currentRecipient, uint256 currentScore, uint256 currentVoteCount, uint256 votesNeeded)
    {
        return (pendingRecipient, pendingScore, voteCount, REQUIRED_VOTES);
    }

    // simplified debug - split into smaller functions to avoid stack issues
    function getDebugBasic()
        external
        view
        returns (
            address lastSender,
            uint256 lastTime,
            bytes32 lastRecipient,
            uint256 totalAttempts
        )
    {
        return (lastAttemptedBy, lastAttemptTime, lastAttemptedRecipient, totalVoteAttempts);
    }

    function getDebugVotes()
        external
        view
        returns (
            bool trustee1Voted,
            bool trustee2Voted,
            bool trustee3Voted
        )
    {
        return (hasVoted[trustee1], hasVoted[trustee2], hasVoted[trustee3]);
    }

    function getDebugStakes()
        external
        view
        returns (
            uint256 ownedStake,
            uint256 totalHotkeyStake_
        )
    {
        return (getOwnedStake(), getTotalHotkeyStake());
    }

    function getDebugIdentity()
        external
        view
        returns (
            bytes32 coldkey_,
            bytes32 hotkey_,
            uint16 netuid_
        )
    {
        return (contractColdkey, contractHotkey, subnet);
    }

    function checkAddressStatus(address addr)
        external
        view
        returns (
            bool isTrusteeAddress,
            bool hasVotedInCurrentRound
        )
    {
        return (isTrustee(addr), hasVoted[addr]);
    }

    function _resetVotes() private {
        pendingRecipient = bytes32(0);
        pendingScore = 0;
        voteCount = 0;
        hasVoted[trustee1] = false;
        hasVoted[trustee2] = false;
        hasVoted[trustee3] = false;
        emit VoteReset();
    }
}