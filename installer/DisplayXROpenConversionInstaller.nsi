; DisplayXR open-model conversion provider installer
; Copyright 2026, The DisplayXR Project
; SPDX-License-Identifier: BSL-1.0
;
; Installs DisplayXR-OpenConversion.dll (+ ONNX Runtime and DirectML, shipped next to it) and
; the default model-pack manifest, and writes the provider registry row under
; HKLM\Software\DisplayXR\ConversionProviders\open-conversion (runtime spec
; conversion-provider-discovery.md sections 3.1 and 3.8). It never downloads a model: the pack
; lists blobs by URL and SHA-256, and the runtime model store fetches them on first use.
;
; Order-independent: no runtime prerequisite. The uninstaller removes only this provider row
; and files, never the runtime or its model store.
;
; Build-time definitions: VERSION, DIST_DIR (a `cmake --install` tree), OUTPUT_DIR.

!ifndef VERSION
	!define VERSION "0.0.0"
!endif
!ifndef DIST_DIR
	!error "pass -DDIST_DIR=<cmake install tree>"
!endif
!ifndef OUTPUT_DIR
	!define OUTPUT_DIR "."
!endif

!define PROVIDER_ID "open-conversion"
!define ROW "Software\DisplayXR\ConversionProviders\${PROVIDER_ID}"
!define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\DisplayXROpenConversion"

Name "DisplayXR Open Conversion ${VERSION}"
OutFile "${OUTPUT_DIR}\DisplayXROpenConversionSetup-${VERSION}.exe"
RequestExecutionLevel admin
InstallDir "$PROGRAMFILES64\DisplayXR\Plugins\OpenConversion"
InstallDirRegKey HKLM "Software\DisplayXR\Plugins\OpenConversion" "InstallPath"
ShowInstDetails show
ShowUninstDetails show
; A locked file aborts the install (non-zero exit) instead of being skipped under a new
; registry Version.
AllowSkipFiles off

!include "x64.nsh"
!include "LogicLib.nsh"

Var MaintMutex

Function .onInit
	${IfNot} ${RunningX64}
		SetErrorLevel 3
		Abort
	${EndIf}
	SetRegView 64
	; Hold the provider maintenance mutex for the whole install, so the service does not
	; respawn the conversion host into a half-written install (spec 3.8 steps 1 and 6).
	System::Call 'kernel32::CreateMutexW(p 0, i 1, w "Global\DisplayXR-ConvMaintenance-${PROVIDER_ID}") p .r0'
	StrCpy $MaintMutex $0
FunctionEnd

Function .onInstSuccess
	${If} $MaintMutex != 0
		System::Call 'kernel32::CloseHandle(p $MaintMutex)'
	${EndIf}
FunctionEnd

Function .onInstFailed
	${If} $MaintMutex != 0
		System::Call 'kernel32::CloseHandle(p $MaintMutex)'
	${EndIf}
FunctionEnd

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Open Conversion provider"
	SectionIn RO
	SetRegView 64
	SetOutPath "$INSTDIR"
	File /r "${DIST_DIR}\*.*"
	WriteUninstaller "$INSTDIR\Uninstall.exe"

	; The row. Values first, Version LAST (the "row complete" marker).
	WriteRegStr   HKLM "${ROW}" "Binary"          "$INSTDIR\DisplayXR-OpenConversion.dll"
	WriteRegStr   HKLM "${ROW}" "DisplayName"     "DisplayXR Open Conversion"
	WriteRegStr   HKLM "${ROW}" "Vendor"          "DisplayXR"
	WriteRegStr   HKLM "${ROW}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
	WriteRegDWORD HKLM "${ROW}" "ProbeOrder"      200
	; Packs is a REG_MULTI_SZ (NSIS has no native writer).
	nsExec::ExecToLog '"$SYSDIR\reg.exe" add "HKLM\${ROW}" /v Packs /t REG_MULTI_SZ /d "$INSTDIR\dxr-open-default.models.json" /f /reg:64'
	Pop $0
	${If} $0 != 0
		SetErrorLevel 6
		Abort "could not write the Packs value"
	${EndIf}
	WriteRegStr   HKLM "${ROW}" "Version"         "${VERSION}"

	WriteRegStr HKLM "Software\DisplayXR\Plugins\OpenConversion" "InstallPath" "$INSTDIR"
	WriteRegStr HKLM "Software\DisplayXR\Plugins\OpenConversion" "Version" "${VERSION}"

	WriteRegStr   HKLM "${UNINST}" "DisplayName"     "DisplayXR Open Conversion"
	WriteRegStr   HKLM "${UNINST}" "DisplayVersion"  "${VERSION}"
	WriteRegStr   HKLM "${UNINST}" "Publisher"       "DisplayXR"
	WriteRegStr   HKLM "${UNINST}" "InstallLocation" "$INSTDIR"
	WriteRegStr   HKLM "${UNINST}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
	WriteRegStr   HKLM "${UNINST}" "QuietUninstallString" "$\"$INSTDIR\Uninstall.exe$\" /S"
	WriteRegDWORD HKLM "${UNINST}" "NoModify" 1
	WriteRegDWORD HKLM "${UNINST}" "NoRepair" 1
SectionEnd

Function un.onInit
	SetRegView 64
	System::Call 'kernel32::CreateMutexW(p 0, i 1, w "Global\DisplayXR-ConvMaintenance-${PROVIDER_ID}") p .r0'
	StrCpy $MaintMutex $0
FunctionEnd

Section "Uninstall"
	SetRegView 64
	; Only this provider row and files. The runtime model store is never touched.
	DeleteRegKey HKLM "${ROW}"
	DeleteRegKey HKLM "Software\DisplayXR\Plugins\OpenConversion"
	DeleteRegKey HKLM "${UNINST}"
	Delete "$INSTDIR\DisplayXR-OpenConversion.dll"
	Delete "$INSTDIR\onnxruntime*.dll"
	Delete "$INSTDIR\DirectML.dll"
	Delete "$INSTDIR\dxr-open-default.models.json"
	RMDir /r "$INSTDIR\licenses"
	Delete "$INSTDIR\Uninstall.exe"
	RMDir "$INSTDIR"
	DeleteRegKey /ifempty HKLM "Software\DisplayXR\Plugins"
	${If} $MaintMutex != 0
		System::Call 'kernel32::CloseHandle(p $MaintMutex)'
	${EndIf}
SectionEnd
