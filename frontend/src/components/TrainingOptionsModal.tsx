/**
 * Training Options Modal Component
 * Training always runs on Modal.com cloud - this modal just confirms.
 */

interface TrainingOptionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (options: {
    backend: 'modal';
  }) => void;
  experimentName?: string;
}

export function TrainingOptionsModal({
  isOpen,
  onClose,
  onConfirm,
  experimentName,
}: TrainingOptionsModalProps) {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '24px',
          maxWidth: '450px',
          width: '90%',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <h2 style={{ marginTop: 0, marginBottom: '16px' }}>
          Run Experiment
        </h2>

        {experimentName && (
          <p style={{ color: '#666', marginBottom: '16px' }}>
            <strong>{experimentName}</strong>
          </p>
        )}

        <p style={{
          fontSize: '14px',
          color: '#1976d2',
          backgroundColor: '#e3f2fd',
          padding: '12px',
          borderRadius: '4px',
        }}>
          Training will run on Modal.com cloud with full resources.
        </p>

        <div style={{
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '12px',
          marginTop: '24px',
          paddingTop: '16px',
          borderTop: '1px solid #eee',
        }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 16px',
              border: '1px solid #ccc',
              borderRadius: '4px',
              backgroundColor: 'white',
              cursor: 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => onConfirm({ backend: 'modal' })}
            style={{
              padding: '8px 16px',
              border: 'none',
              borderRadius: '4px',
              backgroundColor: '#1976d2',
              color: 'white',
              cursor: 'pointer',
            }}
          >
            Run Experiment
          </button>
        </div>
      </div>
    </div>
  );
}
