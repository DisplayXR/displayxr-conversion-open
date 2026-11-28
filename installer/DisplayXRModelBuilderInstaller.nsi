; DisplayXR open model builder installer (research preview)
; Copyright 2026, The DisplayXR Project
; SPDX-License-Identifier: BSL-1.0
;
; Installs dxr-model-builder.exe (a frozen Python program, one folder) and writes its row,
; HKLM\Software\DisplayXR\ModelBuilders\open-model-builder (64-bit view): Binary first, Version
; LAST (runtime spec conversion-provider-discovery.md section 5.6.3 step 3). The runtime's model
; store runs it from `displayxr-cli conv models fetch <pack> --model <name>` (the Dashboard's
; Build button) to build a research-only model ON THIS MACHINE from the official upstream file.
;
; It downloads nothing, builds nothing, never writes AllowResearchOnly or a licence acceptance
; (those are the user's, per account), and holds no model weights, sources or derived files.
; The uninstaller removes only this row and these files, never the model store.
;
; Build-time definitions: VERSION, DIST_DIR (the PyInstaller onedir, dxr-model-builder\),
; OUTPUT_DIR, ALLOWLIST_NSH (optional: the exe's SHA-256 row for an unsigned build, from
; installer/gen_allowlist_nsh.cmake -DFILES=dxr-model-builder.exe) and SIGN_CMD (optional).
;
; Provenance (section 5.6.3 step 4): the store runs only a builder that is Authenticode-signed
; with leaf CN "Leia, Inc." (DXR_FIRST_PARTY_SIGNERS) or allowlisted; a binary signed by anyone
; else is refused even if allowlisted. The caller signs dxr-model-builder.exe (and the .pyd /
; .dll files beside it) BEFORE makensis runs, as for the provider DLL.

!ifndef VERSION
	!define VERSION "0.0.0"
!endif
!ifndef DIST_DIR
	!error "pass -DDIST_DIR=<the dxr-model-builder onedir>"
!endif
!ifndef OUTPUT_DIR
	!define OUTPUT_DIR "."
!endif

; Code signing: same two-pass recipe as DisplayXROpenConversionInstaller.nsi (see there).
!ifndef INNER
	!ifdef SIGN_CMD
		!if "${SIGN_CMD}" != ""
			!ifdef ALLOWLIST_NSH
				!define MB_INNER_ALLOWLIST '"-DALLOWLIST_NSH=${ALLOWLIST_NSH}"'
			!else
				!define MB_INNER_ALLOWLIST ''
			!endif
			!finalize '${SIGN_CMD} "%1"'
			!makensis '-DINNER "-DVERSION=${VERSION}" "-DDIST_DIR=${DIST_DIR}" ${MB_INNER_ALLOWLIST} "DisplayXRModelBuilderInstaller.nsi"' = 0
			!system '"$%TEMP%\DisplayXRModelBuilder_inner.exe"' = 2
			!system '${SIGN_CMD} "$%TEMP%\ModelBuilder_Uninstall.exe"' = 0
			!define USE_PRESIGNED_UNINST
		!endif
	!endif
!endif

!ifdef ALLOWLIST_NSH
	!include "${ALLOWLIST_NSH}"
!else
	!macro DXR_ALLOWLIST_WRITE
	!macroend
	!macro DXR_ALLOWLIST_REMOVE
	!macroend
!endif

!define BUILDER_ID "open-model-builder"
!define ROW "Software\DisplayXR\ModelBuilders\${BUILDER_ID}"
!define UNINST "Software\Microsoft\Windows\CurrentVersion\Uninstall\DisplayXRModelBuilder"

Name "DisplayXR Model Builder (research preview) ${VERSION}"
!ifdef INNER
	OutFile "$%TEMP%\DisplayXRModelBuilder_inner.exe"
	RequestExecutionLevel user
!else
	OutFile "${OUTPUT_DIR}\DisplayXRModelBuilderSetup-${VERSION}.exe"
	RequestExecutionLevel admin
!endif
InstallDir "$PROGRAMFILES64\DisplayXR\ModelBuilder"
ShowInstDetails show
ShowUninstDetails show
AllowSkipFiles off

!include "x64.nsh"
!include "LogicLib.nsh"

Function .onInit
!ifdef INNER
	SetSilent silent
	WriteUninstaller "$%TEMP%\ModelBuilder_Uninstall.exe"
	Quit
!endif
	${IfNot} ${RunningX64}
		SetErrorLevel 3
		Abort
	${EndIf}
	SetRegView 64
FunctionEnd

Page directory
Page instfiles
UninstPage uninstConfirm
UninstPage instfiles

Section "Model builder"
	SectionIn RO
	SetRegView 64
	; An older Version must never describe a half-replaced folder: drop the row's completion
	; marker first, then replace the files.
	DeleteRegValue HKLM "${ROW}" "Version"
	RMDir /r "$INSTDIR\_internal"
	SetOutPath "$INSTDIR"
	File /r "${DIST_DIR}\*.*"
!ifdef USE_PRESIGNED_UNINST
	File "/oname=Uninstall.exe" "$%TEMP%\ModelBuilder_Uninstall.exe"
!else
	WriteUninstaller "$INSTDIR\Uninstall.exe"
!endif

	; The row: Binary first, Version LAST (the "row complete" marker).
	WriteRegStr HKLM "${ROW}" "Binary"          "$INSTDIR\dxr-model-builder.exe"
	WriteRegStr HKLM "${ROW}" "DisplayName"     "DisplayXR Model Builder (research preview)"
	WriteRegStr HKLM "${ROW}" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
	!insertmacro DXR_ALLOWLIST_WRITE
	WriteRegStr HKLM "${ROW}" "Version"         "${VERSION}"

	WriteRegStr   HKLM "${UNINST}" "DisplayName"     "DisplayXR Model Builder (research preview)"
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
FunctionEnd

Section "Uninstall"
	SetRegView 64
	; Only this row and these files: never the model store, AllowResearchOnly or a licence record.
	DeleteRegKey HKLM "${ROW}"
	DeleteRegKey /ifempty HKLM "Software\DisplayXR\ModelBuilders"
	!insertmacro DXR_ALLOWLIST_REMOVE
	DeleteRegKey HKLM "${UNINST}"
	RMDir /r "$INSTDIR\_internal"
	Delete "$INSTDIR\dxr-model-builder.exe"
	Delete "$INSTDIR\Uninstall.exe"
	RMDir "$INSTDIR"
SectionEnd
