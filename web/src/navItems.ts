import { Activity, BarChart3, BookOpen, Boxes, Brain, FileText, Folder, KeyRound, MessagesSquare, Settings, ShieldCheck, Wrench } from 'lucide-react'

export const navItems = [
  { id: 'chat', label: 'Chat', icon: Activity },
  { id: 'sessions', label: 'Sessions', icon: MessagesSquare },
  { id: 'models', label: 'Models', icon: Boxes },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'skills', label: 'Skills', icon: Brain },
  { id: 'diagnostics', label: 'Diagnostics', icon: ShieldCheck },
  { id: 'logs', label: 'Logs', icon: FileText },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'docs', label: 'Docs', icon: BookOpen },
  { id: 'workspace', label: 'Workspace', icon: Folder },
  { id: 'environment', label: 'Environment', icon: KeyRound },
  { id: 'settings', label: 'Settings', icon: Settings },
] as const
