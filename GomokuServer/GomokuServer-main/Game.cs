using System.Text;
using System.Text.Json;

namespace GomokuServer
{
    class PlayerStatistic(string label)
    {
        static readonly JsonSerializerOptions SerializerOptions = new() { WriteIndented = true };

        public string Label { get; } = label;
        public int[] WinCount { get; } = new int[2];
        public int[] LossCount { get; } = new int[2];
        public int[] DrawCount { get; } = new int[2];

        public int TotalWinCount => WinCount.Sum();
        public int TotalLossCount => LossCount.Sum();
        public int TotalDrawCount => DrawCount.Sum();
        public int TotalGameCount => TotalWinCount + TotalLossCount + TotalDrawCount;
        public double TotalWinRate => (TotalWinCount + TotalDrawCount * 0.5) / TotalGameCount;

        public int GameCountWhenBlack => GetGameCountWhen(DiscColor.Black);
        public int GameCountWhenWhite => GetGameCountWhen(DiscColor.White);
        public double WinRateWhenBlack => GetWinRateWhen(DiscColor.Black);
        public double WinRateWhenWhite => GetWinRateWhen(DiscColor.White);

        public int GetGameCountWhen(DiscColor color) => WinCount[(int)color] + LossCount[(int)color] + DrawCount[(int)color];

        public double GetWinRateWhen(DiscColor color)
        {
            var gameCount = GetGameCountWhen(color);
            return (gameCount != 0) ? (WinCount[(int)color] + DrawCount[(int)color] * 0.5) / gameCount : 0.0;
        }

        public void SaveAt(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, SerializerOptions));
    }

    class Player
    {
        public string Name => Engine.Name;
        public PlayerStatistic Stats { get; }
        public Engine Engine { get; }

        public Player(string label, EngineConfig engineConfig)
        => (Stats, Engine) = (new PlayerStatistic(label), new Engine(engineConfig));
    }

    record class GameConfig
    {
        public int BoardSize { get; init; } = 9;
        public int TimeLimitMsPerMove { get; init; } = 5000;
        public bool SwapPlayer { get; init; } = true;
        public bool UseSamePositionWhenSwapPlayer { get; init; } = true;
        public string OpeningPositionsPath { get; init; } = string.Empty;
        public bool ShuffleOpenings { get; init; } = true;
        public string GameLogPath { get; init; } = "game.txt";
        public string GameStatsPath { get; init; } = "stats.json";

        public string ViewerPath { get; init; } = string.Empty;
        public string ViewerArgs { get; init; } = string.Empty;

        static readonly JsonSerializerOptions SaveOptions = new() { WriteIndented = true };

        public static GameConfig? Load(string path) => JsonSerializer.Deserialize<GameConfig>(File.ReadAllText(path));
        public void SaveAt(string path) => File.WriteAllText(path, JsonSerializer.Serialize(this, SaveOptions));
    }

    class Game(GameConfig config, EngineConfig engineConfig0, EngineConfig engineConfig1)
    {
        static readonly JsonSerializerOptions SearializerOptions = new() { WriteIndented = true };

        readonly GameConfig _config = config;
        readonly EngineConfig[] _engineConfigs = [engineConfig0, engineConfig1];

        Viewer? _viewer;
        Action<Position> _showPos = x => { };

        public void Start(int numGames)
        {
            var players = _engineConfigs.Select(x => new Player(x.Name, x)).ToArray();

            for (var i = 0; i < players.Length; i++)
            {
                players[i].Engine.Run();
                if (!players[i].Engine.IsRunning)
                {
                    Console.Error.WriteLine($"Error: Cannot execute \"{_engineConfigs[i].Path}\"");
                    return;
                }
            }

            if (!string.IsNullOrEmpty(_config.ViewerPath))
            {
                _viewer = new Viewer(_config.ViewerPath, _config.ViewerArgs);
                _viewer.Run();
                if (!_viewer.IsRunning)
                {
                    Console.Error.WriteLine($"Error: Cannot execute \"{_config.ViewerPath}\"");
                    return;
                }
                _showPos = x => _viewer.SetPosition(x);
            }
            else
            {
                _showPos = x => { };
            }

            Position[]? openings = LoadOpenings(_config.OpeningPositionsPath);

            if (openings is null)
                return;

            if (!Mainloop(numGames, openings, players))
                Console.Error.WriteLine("Error: Game was suspended.");

            QuitEngines(players);
        }

        bool Mainloop(int numGames, Position[] openings, Player[] players)
        {
            using var gameLog = new StreamWriter(_config.GameLogPath);

            var pos = new Position(_config.BoardSize);
            var openingIdx = 0;
            Random.Shared.Shuffle(openings);
            for (var gameID = 0; gameID < numGames; gameID++)
            {
                if (!_config.SwapPlayer || !_config.UseSamePositionWhenSwapPlayer || gameID % 2 == 0)
                {
                    openings[openingIdx++].CopyTo(pos);
                    if (openingIdx == openings.Length)
                    {
                        Random.Shared.Shuffle(openings);
                        openingIdx = 0;
                    }
                }

                Console.WriteLine($"Game {gameID + 1}");

                if (!PlayOneGame(pos, (_config.SwapPlayer && gameID % 2 == 1) ? [.. players.Reverse()] : players, gameLog))
                    return false;

                Console.WriteLine("////////////////////");
                foreach ((var engine, var stats) in players.Select(x => (x.Engine, x.Stats)))
                    Console.WriteLine($"{engine.Name}: {stats.TotalWinCount}-{stats.TotalDrawCount}-{stats.TotalLossCount} (WinRate: {stats.TotalWinRate * 100.0}%)");
                Console.WriteLine("////////////////////");

                SaveStats(players);
            }

            return true;
        }

        bool PlayOneGame(Position rootPos, Player[] players, StreamWriter gameLog)
        {
            foreach (var p in players)
                p.Engine.SetPosition(rootPos);

            Player player, opponent;
            var pos = new Position(rootPos);
            var moves = new List<int>();
            while (true)
            {
                (player, opponent) = (players[(int)pos.SideToMove], players[(int)pos.OpponentColor]);
                var move = player.Engine.Think(_config.TimeLimitMsPerMove);

                if (move == -1)
                {
                    Console.Error.WriteLine($"Error: \"{player.Name}\" returned invalid coordinate.");
                    return false;
                }

                if (!pos.Update(move))
                {
                    Console.Error.WriteLine($"Error: move {pos.CoordinateToString(move)} which played by \"{player.Name}\" is illegal");
                    return false;
                }

                _showPos(pos);

                moves.Add(move);

                opponent.Engine.SendMove(move);

                if (pos.Winner != DiscColor.Null || pos.IsFull)
                {
                    _viewer?.SendGameResult(pos.Winner);

                    if (pos.Winner == DiscColor.Null)
                        {
                            Console.WriteLine("Gameover: Draw");

                            for (var i = 0; i < players.Length; i++)
                                players[i].Stats.DrawCount[i]++;
                            break;
                        }

                    Console.WriteLine($"Gameover: {players[(int)pos.Winner].Engine.Name} wins");

                    players[(int)pos.Winner].Stats.WinCount[(int)pos.Winner]++;
                    players[(int)pos.Loser].Stats.LossCount[(int)pos.Loser]++;
                    break;
                }
            }

            gameLog.WriteLine($"{rootPos} {string.Join("", moves)}");

            return true;
        }

        void SaveStats(Player[] players) => File.WriteAllText(_config.GameStatsPath, JsonSerializer.Serialize(players, SearializerOptions));

        Position[]? LoadOpenings(string path)
        {
            if (string.IsNullOrEmpty(path))
                return [new Position(_config.BoardSize)];

            var positions = new List<Position>();
            using var sr = new StreamReader(path);
            var lineCount = 0;
            while (sr.Peek() != -1)
            {
                var line = sr.ReadLine();

                if (line is null)
                    continue;

                lineCount++;
                var pos = Position.TryParse(line);

                if (pos is null)
                {
                    Console.Error.WriteLine($"Error: A position at line {lineCount} is invalid.");
                    return null;
                }

                if (pos.BoardSize != _config.BoardSize)
                {
                    Console.Error.WriteLine($"Warning: The board size of position at line {lineCount} is {pos.BoardSize} but the server expected {_config.BoardSize}.");
                    return null;
                }

                positions.Add(pos);
            }
            return [.. positions];
        }

        void QuitEngines(Player[] players)
        {
            const int TimeoutMs = 10000;
            foreach (var player in players)
            {
                player.Engine.Quit(TimeoutMs);
                if (player.Engine.IsRunning)
                    player.Engine.Kill();
            }

            _viewer?.Quit(TimeoutMs);
            if (_viewer is not null && _viewer.IsRunning)
                _viewer.Kill();
        }
    }
}