!include "MUI2.nsh"
!ifndef DIST_DIR
  !define DIST_DIR "dist"
!endif
Name "Agrista"
OutFile "${DIST_DIR}\Agrista-Setup.exe"
InstallDir "$PROGRAMFILES64\Agrista"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "Turkish"

Section "Kurulum"
  SetOutPath "$INSTDIR"
  File /r "${DIST_DIR}\Agrista\*.*"
  CreateShortcut "$SMPROGRAMS\Agrista.lnk" "$INSTDIR\Agrista.exe"
  CreateShortcut "$DESKTOP\Agrista.lnk" "$INSTDIR\Agrista.exe"
  WriteUninstaller "$INSTDIR\Kaldir.exe"
SectionEnd

Section "Kaldırıcı"
  Delete "$INSTDIR\Kaldir.exe"
  RMDir /r "$INSTDIR"
SectionEnd
