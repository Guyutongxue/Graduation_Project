﻿// See https://aka.ms/new-console-template for more information
using FlaUI.UIA3;
using FlaUI.Core;
using System.Diagnostics;
using FlaUI.Core.AutomationElements;
using HttpJsonRpc;
using System.Net;

namespace formcheck
{
  [JsonRpcClass("formcheck")]
  class Handler
  {
    [JsonRpcMethod]
    void Add()
    {
      var TEST_EXE = @"C:\Users\guyutongxue\Documents\MyFiles\Code\Csharp\WeiboFishHack\WeiboFishHack\bin\Debug\net7.0-windows\WeiboFishHack.exe";

      Program.Check(TEST_EXE);
    }
  }

  class Program
  {
    static public void Check(string exePath)
    {
      Application? app = null;
      try
      {
        Console.WriteLine("Hello, World!");
        app = Application.Launch(exePath);
        using var automation = new UIA3Automation();
        var window = app.GetMainWindow(automation);
        var elements = window.FindAllDescendants();
        var button = window.FindFirstDescendant(cf => cf.ByText("发送"));
        if (button is null)
        {
          throw new Exception("null button1");
        }
        button.Click();
        Thread.Sleep(1000);
        var textarea = window.FindFirstDescendant(cf => cf.ByAutomationId("textBox1"));
        Console.WriteLine(textarea.AsTextBox().Text);
        Console.WriteLine(window.Title);
      }
      finally
      {
        app?.Close();
      }
    }
    static void Main(String[] args)
    {
      if (args.Length < 1 || !int.TryParse(args[1], out var port)) {
        throw new Exception("Port not number");
      }
      JsonRpc.ServerOptions = (o) =>
      {
        o.Listen(new IPEndPoint(IPAddress.Loopback, port));
      };
      JsonRpc.Start();

    }
  }
}


