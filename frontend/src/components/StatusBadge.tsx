type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info';

interface StatusBadgeProps {
  status: string;
  variant?: BadgeVariant;
}

const statusVariants: Record<string, BadgeVariant> = {
  // Project statuses
  draft: 'default',
  active: 'success',
  archived: 'warning',

  // Experiment/Trial statuses
  pending: 'default',
  running: 'info',
  completed: 'success',
  failed: 'error',
  cancelled: 'warning',

  // Model statuses
  candidate: 'default',
  shadow: 'info',
  production: 'success',
  retired: 'warning',
};

export default function StatusBadge({ status, variant }: StatusBadgeProps) {
  const badgeVariant = variant || statusVariants[status] || 'default';

  return (
    <span className={`status-badge status-badge-${badgeVariant}`}>
      {status}
    </span>
  );
}
