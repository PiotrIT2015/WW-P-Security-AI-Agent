/*
 Secure RAG YARA Rules

 Detection:
 - suspicious scripts
 - prompt injection
 - malware indicators
*/


rule Suspicious_PowerShell
{

    strings:

        $ps1 =
            "powershell"

        $ps2 =
            "Invoke-WebRequest"

        $ps3 =
            "DownloadString"


    condition:

        any of them

}



rule Prompt_Injection_Attempt
{

    strings:

        $inject1 =
            "ignore previous instructions"

        $inject2 =
            "system prompt"

        $inject3 =
            "developer message"

        $inject4 =
            "reveal your instructions"


    condition:

        any of them

}



rule Suspicious_Script_Content
{

    strings:

        $cmd =
            "cmd.exe"

        $bash =
            "/bin/bash"

        $python =
            "os.system"


    condition:

        any of them

}