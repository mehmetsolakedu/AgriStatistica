!include "MUI2.nsh"
Name "Agrista"
OutFile "dist\Agrista-Setup.exe"
InstallDir "$PROGRAMFILES64\Agrista"
RequestExecutionLevel admin

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "Turkish"

Section "Kurulum"
  SetOutPath "$INSTDIR"
  File /r "dist\Agrista\*.*"
  CreateShortcut "$SMPROGRAMS\Agrista.lnk" "$INSTDIR\Agrista.exe"
  CreateShortcut "$DESKTOP\Agrista.lnk" "$INSTDIR\Agrista.exe"
  WriteUninstaller "$INSTDIR\Kaldir.exe"
SectionEnd

Section "Kaldırıcı"
  Delete "$INSTDIR\Kaldir.exe"
  RMDir /r "$INSTDIR"
SectionEnd
