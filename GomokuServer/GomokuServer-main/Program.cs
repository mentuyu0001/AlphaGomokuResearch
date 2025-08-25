using System.Text.Json;
using GomokuServer;

const int NumArgs = 4;


#if DEBUG
Environment.CurrentDirectory = "../../../DebugWorkDir";
args = ["game_config.json", "random_engine.json", "random_engine.json", "1"];
#endif

if (args.Length < NumArgs)
{
    Console.WriteLine("Hint");
    Console.WriteLine("args[0]: game config json path");
    Console.WriteLine("args[1]: engine config json path");
    Console.WriteLine("args[2]: engine config json path");
    Console.WriteLine("args[3]: the number of games");
    Console.WriteLine();
    Console.WriteLine("Example: Server.exe game_config.json engine_config_0.json engine_config_1.json 100");
    return;
}

var gameConfig = JsonSerializer.Deserialize<GameConfig>(File.ReadAllText(args[0]));

if (gameConfig is null)
{
    Console.Error.WriteLine($"Invalid file: \"{args[0]}\"");
    return;
}

var engineConfig0 = JsonSerializer.Deserialize<EngineConfig>(File.ReadAllText(args[1]));

if (engineConfig0 is null)
{
    Console.Error.WriteLine($"Invalid file: \"{args[1]}\"");
    return;
}

var engineConfig1 = JsonSerializer.Deserialize<EngineConfig>(File.ReadAllText(args[2]));

if (engineConfig1 is null)
{
    Console.Error.WriteLine($"Invalid file: \"{args[2]}\"");
    return;
}

if (!int.TryParse(args[3], out int numGames))
{
    Console.Error.WriteLine($"The number of game is invalid. It must be a positive integer.");
    return;
}

var game = new Game(gameConfig, engineConfig0, engineConfig1);
game.Start(numGames);