import { useState } from 'react';
import { explainModel, ApiException } from '../services/api';

interface ModelExplainerProps {
  modelId: string;
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  modelName?: string; // Reserved for future use
}

const EXAMPLE_QUESTIONS = [
  'Which features are most important?',
  'What are the model\'s strengths and weaknesses?',
  'How can I improve the model performance?',
  'Explain the model metrics in plain language.',
];

export default function ModelExplainer({ modelId }: ModelExplainerProps) {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await explainModel(modelId, question);
      setAnswer(result.answer);
    } catch (err) {
      if (err instanceof ApiException) {
        setError(err.detail);
      } else {
        setError('Failed to get explanation');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (example: string) => {
    setQuestion(example);
    setAnswer(null);
    setError(null);
  };

  return (
    <div className="model-explainer">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="question">Ask a question about your model</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., Which features are most important?"
            rows={3}
            className="form-textarea"
            style={{ width: '100%', resize: 'vertical' }}
          />
        </div>

        <div className="example-questions" style={{ marginBottom: '1rem' }}>
          <span style={{ fontSize: '0.875rem', color: '#6b7280', marginRight: '0.5rem' }}>
            Try:
          </span>
          {EXAMPLE_QUESTIONS.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => handleExampleClick(example)}
              className="btn btn-small"
              style={{
                margin: '0.25rem',
                padding: '0.25rem 0.5rem',
                fontSize: '0.75rem',
                backgroundColor: '#f3f4f6',
                border: '1px solid #e5e7eb',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              {example}
            </button>
          ))}
        </div>

        {error && (
          <div className="form-error" style={{ marginBottom: '1rem' }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary"
          disabled={isLoading || !question.trim()}
        >
          {isLoading ? (
            <>
              <span className="spinner spinner-small"></span>
              Analyzing...
            </>
          ) : (
            <>
              <span className="btn-icon">💡</span>
              Get Explanation
            </>
          )}
        </button>
      </form>

      {answer && (
        <div className="explanation-result" style={{ marginTop: '1.5rem' }}>
          <div className="detail-card" style={{
            backgroundColor: '#fefce8',
            border: '1px solid #fef08a',
          }}>
            <h4 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span style={{ fontSize: '1.25rem' }}>🤖</span>
              AI Explanation
            </h4>
            <div className="markdown-content" style={{
              lineHeight: 1.6,
              color: '#374151',
              whiteSpace: 'pre-wrap',
            }}>
              {answer}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
