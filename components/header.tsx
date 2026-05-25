'use client';

import {
  Settings,
  Sun,
  Moon,
  Monitor,
  ArrowLeft,
  Loader2,
  Download,
  FileDown,
  Package,
  Archive,
  Library,
} from 'lucide-react';
import { useI18n } from '@/lib/hooks/use-i18n';
import { useTheme } from '@/lib/hooks/use-theme';
import { LanguageSwitcher } from './language-switcher';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { SettingsDialog } from './settings';
import { cn } from '@/lib/utils';
import { useStageStore } from '@/lib/store/stage';
import { useMediaGenerationStore } from '@/lib/store/media-generation';
import { useExportPPTX } from '@/lib/export/use-export-pptx';
import { useExportClassroom } from '@/lib/export/use-export-classroom';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { RagEvidencePanel } from '@/components/knowledge/rag-evidence-panel';
import type { RagEvidence } from '@/lib/types/rag';

interface HeaderProps {
  readonly currentSceneTitle: string;
}

export function Header({ currentSceneTitle }: HeaderProps) {
  const { t } = useI18n();
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [themeOpen, setThemeOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidence, setEvidence] = useState<RagEvidence | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);
  const [evidenceLoading, setEvidenceLoading] = useState(false);

  // Export
  const { exporting: isExporting, exportPPTX, exportResourcePack } = useExportPPTX();
  const { exporting: isExportingZip, exportClassroomZip } = useExportClassroom();
  const [exportMenuOpen, setExportMenuOpen] = useState(false);
  const exportRef = useRef<HTMLDivElement>(null);
  const scenes = useStageStore((s) => s.scenes);
  const ragSnapshotId = useStageStore((s) => s.stage?.ragSnapshotId);
  const generatingOutlines = useStageStore((s) => s.generatingOutlines);
  const failedOutlines = useStageStore((s) => s.failedOutlines);
  const mediaTasks = useMediaGenerationStore((s) => s.tasks);

  const canExport =
    scenes.length > 0 &&
    generatingOutlines.length === 0 &&
    failedOutlines.length === 0 &&
    Object.values(mediaTasks).every((task) => task.status === 'done' || task.status === 'failed');

  const themeRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  const handleClickOutside = useCallback(
    (e: MouseEvent) => {
      if (themeOpen && themeRef.current && !themeRef.current.contains(e.target as Node)) {
        setThemeOpen(false);
      }
      if (exportMenuOpen && exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportMenuOpen(false);
      }
    },
    [themeOpen, exportMenuOpen],
  );

  useEffect(() => {
    if (themeOpen || exportMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [themeOpen, exportMenuOpen, handleClickOutside]);

  const openEvidence = async () => {
    if (!ragSnapshotId) return;
    setEvidenceOpen(true);
    if (evidence?.id === ragSnapshotId) return;
    setEvidenceLoading(true);
    setEvidenceError(null);
    try {
      const response = await fetch(`/api/knowledge/snapshots/${encodeURIComponent(ragSnapshotId)}`);
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || '无法读取检索依据');
      }
      setEvidence(data.evidence);
    } catch (error) {
      setEvidenceError(error instanceof Error ? error.message : '无法读取检索依据');
    } finally {
      setEvidenceLoading(false);
    }
  };

  return (
    <>
      <header className="h-20 px-8 flex items-center justify-between z-10 bg-transparent gap-4">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <button
            onClick={() => router.push('/')}
            className="shrink-0 p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            title={t('generation.backToHome')}
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex flex-col min-w-0">
            <span className="text-[10px] uppercase tracking-widest font-bold text-gray-400 dark:text-gray-500 mb-0.5">
              {t('stage.currentScene')}
            </span>
            <h1
              className="text-xl font-bold text-gray-800 dark:text-gray-200 tracking-tight truncate"
              suppressHydrationWarning
            >
              {currentSceneTitle || t('common.loading')}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-4 bg-white/60 dark:bg-gray-800/60 backdrop-blur-md px-2 py-1.5 rounded-full border border-gray-100/50 dark:border-gray-700/50 shadow-sm shrink-0">
          {/* Language Selector */}
          <LanguageSwitcher onOpen={() => setThemeOpen(false)} />

          <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700" />

          {/* Theme Selector */}
          <div className="relative" ref={themeRef}>
            <button
              onClick={() => {
                setThemeOpen(!themeOpen);
              }}
              className="p-2 rounded-full text-gray-400 dark:text-gray-500 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm transition-all group"
            >
              {theme === 'light' && <Sun className="w-4 h-4" />}
              {theme === 'dark' && <Moon className="w-4 h-4" />}
              {theme === 'system' && <Monitor className="w-4 h-4" />}
            </button>
            {themeOpen && (
              <div className="absolute top-full mt-2 right-0 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50 min-w-[140px]">
                <button
                  onClick={() => {
                    setTheme('light');
                    setThemeOpen(false);
                  }}
                  className={cn(
                    'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                    theme === 'light' &&
                      'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                  )}
                >
                  <Sun className="w-4 h-4" />
                  {t('settings.themeOptions.light')}
                </button>
                <button
                  onClick={() => {
                    setTheme('dark');
                    setThemeOpen(false);
                  }}
                  className={cn(
                    'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                    theme === 'dark' &&
                      'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                  )}
                >
                  <Moon className="w-4 h-4" />
                  {t('settings.themeOptions.dark')}
                </button>
                <button
                  onClick={() => {
                    setTheme('system');
                    setThemeOpen(false);
                  }}
                  className={cn(
                    'w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2',
                    theme === 'system' &&
                      'bg-purple-50 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
                  )}
                >
                  <Monitor className="w-4 h-4" />
                  {t('settings.themeOptions.system')}
                </button>
              </div>
            )}
          </div>

          <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700" />

          {/* Settings Button */}
          {ragSnapshotId && (
            <>
              <button
                onClick={openEvidence}
                className="p-2 rounded-full text-emerald-600 dark:text-emerald-400 hover:bg-white dark:hover:bg-gray-700 hover:shadow-sm transition-all"
                title="查看本课参考材料"
              >
                <Library className="w-4 h-4" />
              </button>
              <div className="w-[1px] h-4 bg-gray-200 dark:bg-gray-700" />
            </>
          )}

          <div className="relative">
            <button
              onClick={() => setSettingsOpen(true)}
              className="p-2 rounded-full text-gray-400 dark:text-gray-500 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm transition-all group"
            >
              <Settings className="w-4 h-4 group-hover:rotate-90 transition-transform duration-500" />
            </button>
          </div>
        </div>

        {/* Export Dropdown */}
        <div className="relative" ref={exportRef}>
          <button
            onClick={() => {
              if (canExport && !isExporting && !isExportingZip) setExportMenuOpen(!exportMenuOpen);
            }}
            disabled={!canExport || isExporting || isExportingZip}
            title={
              canExport
                ? isExporting || isExportingZip
                  ? t('export.exporting')
                  : t('export.pptx')
                : t('share.notReady')
            }
            className={cn(
              'shrink-0 p-2 rounded-full transition-all',
              canExport && !isExporting && !isExportingZip
                ? 'text-gray-400 dark:text-gray-500 hover:bg-white dark:hover:bg-gray-700 hover:text-gray-800 dark:hover:text-gray-200 hover:shadow-sm'
                : 'text-gray-300 dark:text-gray-600 cursor-not-allowed opacity-50',
            )}
          >
            {isExporting || isExportingZip ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
          </button>
          {exportMenuOpen && (
            <div className="absolute top-full mt-2 right-0 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden z-50 min-w-[200px]">
              <button
                onClick={() => {
                  setExportMenuOpen(false);
                  exportPPTX();
                }}
                className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2.5"
              >
                <FileDown className="w-4 h-4 text-gray-400 shrink-0" />
                <span>{t('export.pptx')}</span>
              </button>
              <button
                onClick={() => {
                  setExportMenuOpen(false);
                  exportResourcePack();
                }}
                className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2.5"
              >
                <Package className="w-4 h-4 text-gray-400 shrink-0" />
                <div>
                  <div>{t('export.resourcePack')}</div>
                  <div className="text-[11px] text-gray-400 dark:text-gray-500">
                    {t('export.resourcePackDesc')}
                  </div>
                </div>
              </button>
              <button
                onClick={() => {
                  setExportMenuOpen(false);
                  exportClassroomZip();
                }}
                disabled={isExportingZip}
                className="w-full px-4 py-2.5 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center gap-2.5"
              >
                <Archive className="w-4 h-4 text-gray-400 shrink-0" />
                <div>
                  <div>{t('export.classroomZip')}</div>
                  <div className="text-[11px] text-gray-400 dark:text-gray-500">
                    {t('export.classroomZipDesc')}
                  </div>
                </div>
              </button>
            </div>
          )}
        </div>
      </header>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      <Dialog open={evidenceOpen} onOpenChange={setEvidenceOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>本课参考材料</DialogTitle>
            <DialogDescription>以下材料片段用于约束课程内容与教师讲解。</DialogDescription>
          </DialogHeader>
          {evidenceLoading ? (
            <div className="py-12 text-center text-sm text-muted-foreground">
              正在读取检索依据...
            </div>
          ) : evidenceError ? (
            <div className="rounded-md border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
              {evidenceError}
            </div>
          ) : evidence ? (
            <RagEvidencePanel evidence={evidence} />
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
