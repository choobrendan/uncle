import React, { useState, useEffect, useRef } from 'react';
import './Community.css';

const LANGUAGES = {
  en: {
    tasks: [
      {
        id: 'yoga',
        description: 'Locate the Yoga Workshop Details',
        hints: [
          'Find the suitable icon in the navigation bar',
          'Click on the box that has the yoga details.',
        ],
      },
      {
        id: 'art',
        description: 'Find and Register for the Watercolor Painting Class',
        hints: [
          'Go to the classes tab',
          'Select the watercolour painting class',
          'The full name and email have been filled up for you.',
        ],
      },
      {
        id: 'facility',
        description: 'Locate the Wheelchair-Accessible Restroom Information',
        hints: [
          'Navigate to the location section',
          'Locate and click on the accessibility features.',
        ],
      },
      {
        id: 'volunteer',
        description: 'Sign Up for Community Volunteering Opportunities',
        hints: [
          'Go to the about tab',
          'Click the sign up via the volunteering opportunities',
          'The details have been filled up for you.',
        ],
        additionalContext: 'We need helpers for events and daily operations. Background check required.',
      },
      {
        id: 'donation',
        description: 'Make a Virtual Donation to the Center',
        hints: [
          'Go to the support section',
          'Click on the donate now button',
          'The details and a fake card number have been filled up.',
        ],
      },
      {
        id: 'bookclub',
        description: 'Join the Senior Fiction Book Discussion Group',
        hints: [
          'Check the navigation bar',
          'Click on the box that has the book club details.',
        ],
        additionalContext: 'Monthly literary discussion group meeting in the community library.',
      },
    ],
    events: [
      {
        category: 'Wellness',
        events: [
          {
            name: 'Gentle Yoga for Seniors',
            description: 'Low-impact yoga for mobility and relaxation',
            time: 'Tuesdays 10:00 AM',
            location: 'Wellness Studio',
          },
          {
            name: 'Meditation Workshop',
            description: 'Mindfulness techniques for stress relief',
            time: 'Thursdays 2:00 PM',
            location: 'Quiet Room',
          },
        ],
      },
      {
        category: 'Arts & Crafts',
        events: [
          {
            name: 'Watercolor Painting Class',
            description: 'Explore watercolor techniques for beginners',
            time: 'Wednesdays 1:00 PM',
            location: 'Art Studio',
          },
          {
            name: 'Knitting Circle',
            description: 'Social knitting and craft sharing',
            time: 'Mondays 11:00 AM',
            location: 'Community Room',
          },
        ],
      },
      {
        category: 'Special Programs',
        events: [
          {
            name: 'Senior Fiction Book Club',
            description: 'Monthly literary discussion group',
            time: 'Last Friday 3:00 PM',
            location: 'Community Library',
          },
          {
            name: 'Community Volunteer Orientation',
            description: 'Learn about volunteering opportunities',
            time: 'First Monday 10:00 AM',
            location: 'Main Hall',
          },
        ],
      },
    ],
    navBar:[
      { icon: '🏠', name: 'home', label: 'Home' },
      { icon: '📅', name: 'events', label: 'Events' },
      { icon: '🎨', name: 'classes', label: 'Classes' },
      { icon: '❤️', name: 'support', label: 'Support' },
      { icon: '👥', name: 'about', label: 'About' },
      { icon: '📞', name: 'contact', label: 'Contact' },
      { icon: '📍', name: 'map', label: 'Location' }
                ],
    currentTask:"Current Task",
    welcome:"Welcome to Community Center",
    discover:"Discover programs, classes, and resources designed for our community.",
    latestNews:"Latest News",
    latestNewsDesc:"New wellness programs starting this month!",
    upcomingEvents:"Upcoming Events",
    upcomingEventsDesc:"Check out our exciting community activities.",
    upcomingEventsTitle:"Upcoming Events and Workshops",
    classRegistration:"Class Registration",
    selectClass:"Select Class",
    registerClass:"Register for Class",
    contact:"Contact & Accessibility",
    support:"Support The Centre",
    makeDonation:"Make a Donation",
    donateNow:"Donate Now",
    aboutUs:"About Us",
    volunteerOpportunities:"Volunteer Opportunities",
    signUpVolunteer:"Sign Up Volunteer",
    parking:"Parking Accessibility",
    parkingAccessibility: "Special parking arrangements available for mobility-impaired visitors. Please inquire at front desk for detailed information.",
    address: "Address: 123 Community Lane",
    phone: "Phone: +6012-345-6789",
    email: "Email: accessibility@communitycenter.org",
    centerLocation: "Center Location & Accessibility",
    accessibilityFeatures:"Accessibility Features",
    wheelchairAccessibleEntrance: "Wheelchair Accessible Entrance",
    elevatorAccess: "Elevator Access",
    handicapParkingAvailable: "Handicap Parking Available"
  },
  zh: {
    tasks: [
      {
        id: 'yoga',
        description: '查找瑜伽工作坊详情',
        hints: ['检查导航栏', '点击包含瑜伽详情的框。'],
      },
      {
        id: 'art',
        description: '查找并注册水彩绘画课程',
        hints: ['转到课程标签', '选择水彩绘画课程', '您的全名和电子邮件已为您填写。'],
      },
      {
        id: 'facility',
        description: '查找无障碍洗手间信息',
        hints: ['检查位置部分', '找到并点击无障碍功能。'],
      },
      {
        id: 'volunteer',
        description: '注册社区志愿者机会',
        hints: ['转到关于标签', '通过志愿者机会点击注册', '详细信息已为您填写。'],
        additionalContext: '我们需要活动助手和日常运营人员。需进行背景调查。',
      },
      {
        id: 'donation',
        description: '向中心进行虚拟捐赠',
        hints: ['转到支持部分', '点击立即捐赠按钮', '详细信息和一个虚拟卡号已填写。'],
      },
      {
        id: 'bookclub',
        description: '加入老年小说书籍讨论组',
        hints: ['检查导航栏', '点击包含书籍俱乐部详情的框。'],
        additionalContext: '每月在社区图书馆举行的文学讨论组。',
      },
    ],
    events: [
      {
        category: '健康',
        events: [
          {
            name: '老年人轻柔瑜伽',
            description: '低强度瑜伽，增强活动能力和放松',
            time: '每周二 10:00 AM',
            location: '健康工作室',
          },
          {
            name: '冥想工作坊',
            description: '缓解压力的正念技巧',
            time: '每周四 2:00 PM',
            location: '安静室',
          },
        ],
      },
      {
        category: '艺术与手工艺',
        events: [
          {
            name: '水彩绘画课程',
            description: '初学者水彩技巧探索',
            time: '每周三 1:00 PM',
            location: '艺术工作室',
          },
          {
            name: '编织圈',
            description: '社交编织和手工艺分享',
            time: '每周一 11:00 AM',
            location: '社区室',
          },
        ],
      },
      {
        category: '特别项目',
        events: [
          {
            name: '老年小说书籍俱乐部',
            description: '每月文学讨论组',
            time: '每月最后一个周五 3:00 PM',
            location: '社区图书馆',
          },
          {
            name: '社区志愿者培训',
            description: '了解志愿者机会',
            time: '每月第一个周一 10:00 AM',
            location: '主厅',
          },
        ],
      },
    ],
    navBar:[
      { icon: '🏠', label: '家', name: 'home' },
      { icon: '📅', label: '事件', name: 'events' },
      { icon: '🎨', label: '课程', name: 'classes' },
      { icon: '❤️', label: '支持', name: 'support' },
      { icon: '👥', label: '关于', name: 'about' },
      { icon: '📞', label: '联系', name: 'contact' },
      { icon: '📍', label: '位置', name: 'location' }
    ],
    currentTask: "当前任务",
    welcome: "欢迎来到社区中心",
    discover: "探索为我们的社区设计的项目、课程和资源。",
    latestNews: "最新消息",
    latestNewsDesc: "本月新推出的健康计划！",
    upcomingEvents: "即将举行的活动",
    upcomingEventsDesc: "查看我们令人兴奋的社区活动。",
    upcomingEventsTitle: "即将举行的活动和研讨会",
    classRegistration: "课程注册",
    selectClass: "选择课程",
    registerClass: "注册课程",
    contact: "联系与无障碍服务",
    support: "支持中心",
    makeDonation: "进行捐赠",
    donateNow: "立即捐赠",
    aboutUs: "关于我们",
    volunteerOpportunities: "志愿者机会",
    signUpVolunteer: "注册志愿者",
    parking:"停車處",
    parkingAccessibility: "为行动不便的访客提供特殊停车安排。请向前台咨询详细信息。",
    address: "地址：123 社区路",
    phone: "电话：+6012-345-6789",
    email: "电子邮件：accessibility@communitycenter.org",
    centerLocation: "中心位置与无障碍服务",
    accessibilityFeatures: "无障碍功能",
    wheelchairAccessibleEntrance: "轮椅无障碍入口",
    elevatorAccess: "电梯通道",
    handicapParkingAvailable: "提供残疾人停车位",
  },
};

const CommunityCenter = ({setNextGame}) => {
  const [currentTask, setCurrentTask] = useState(null);
  const [activeSection, setActiveSection] = useState('Home');
  const [taskComplete, setTaskComplete] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [interactionLog, setInteractionLog] = useState([]);
  const [taskStartTime, setTaskStartTime] = useState(null);
  const searchInputRef = useRef(null);
  const [selectedLanguage, setSelectedLanguage] = useState('en');
const nextGame=()=>{
  setNextGame(2)
}
  useEffect(() => {
    const randomTask = LANGUAGES[selectedLanguage].tasks[Math.floor(Math.random() * LANGUAGES[selectedLanguage].tasks.length)];
    setCurrentTask(randomTask);
    setTaskComplete(false);
    setInteractionLog([]);
    setTaskStartTime(Date.now());
  }, []);

  const logInteraction = (type, details) => {
    const newLog = [
      ...interactionLog,
      { 
        timestamp: Date.now() - taskStartTime, 
        type, 
        details 
      }
    ];
    setInteractionLog(newLog);
  };

  const handleTaskComplete = () => {
    const endTime = Date.now() - taskStartTime;
    setTaskComplete(true);
    logInteraction('task_complete', {
      taskId: currentTask.id,
      duration: endTime
    });
  };

  const renderSection = () => {
    switch(activeSection) {
      case 'events':
        return (
          <div className="section">
            <h2>{LANGUAGES[selectedLanguage].upcomingEventsTitle}</h2>
            {LANGUAGES[selectedLanguage].events.map((category, index) => (
              <div key={index}>
                <h3>{category.category}</h3>
                <div style={{display:"flex",flexDirection:"row",justifyContent:"space-between"}}>
                {category.events.map((event, eventIndex) => (
                  <div  style={{width: "335px"}}
                    key={eventIndex} 
                    className="section-item"
                    onClick={() => {
                      logInteraction('event_click', { event: event.name });
                      if (currentTask.id === 'yoga' && event.name.includes('Yoga')) {
                        handleTaskComplete();
                      }
                      else if (currentTask.id === 'bookclub' && event.name.includes('Book Club')) {
                        handleTaskComplete();
                      }
                    }}
                  >
                    <h4>{event.name}</h4>
                    <p>{event.description}</p>
                    <p className="text-small">{event.time} | {event.location}</p>
                  </div>
                ))}
                </div>
              </div>
            ))}
          </div>
        );
      case 'classes':
        return (
          <div className="section">
            <h2>{LANGUAGES[selectedLanguage].classRegistration}</h2>
            <form 
              className="form"
              onSubmit={(e) => {
                e.preventDefault();
                logInteraction('form_submit', { form: 'class_registration' });
                if (currentTask.id === 'art') {
                  handleTaskComplete();
                }
              }}
            >
              <input 
                type="text" 
                 value="Dohn Joe"
                placeholder="Full Name" 
                required
              />
              <input 
                type="email" 
                value="email@gmail.com"
                placeholder="Email Address" 
                required
              />
              <select 
                onChange={(e) => logInteraction('class_selection', { class: e.target.value })}
              >
                <option>{LANGUAGES[selectedLanguage].selectClass}</option>
                {LANGUAGES[selectedLanguage].events.flatMap(category => 
                  category.events.map(event => (
                    <option key={event.name} value={event.name}>
                      {event.name}
                    </option>
                  ))
                )}
              </select>
              <button type="submit">
              {LANGUAGES[selectedLanguage].registerClass}
              </button>
            </form>
          </div>
        );
      case 'contact':
        return (
          <div className="section">
            <h2>{LANGUAGES[selectedLanguage].contact}</h2>
            <div 
              className="section-item"
              onClick={() => {
                logInteraction('contact_detail', { section: 'parking' });
                if (currentTask.id === 'contact') {
                  handleTaskComplete();
                }
              }}
            >
              <h3>{LANGUAGES[selectedLanguage].parking}</h3>
              <p>
              {LANGUAGES[selectedLanguage].parkingAccessibility}
              </p>
            </div>
            <div className="text-small">
              <p>{LANGUAGES[selectedLanguage].address}</p>
              <p>{LANGUAGES[selectedLanguage].phone}</p>
              <p>{LANGUAGES[selectedLanguage].email}</p>
            </div>
          </div>
        );
      case 'map':
        return (
          <div className="section">
            <h2>{LANGUAGES[selectedLanguage].centerLocation}</h2>
            <div 
              className="map-interaction"
              onClick={() => {
                logInteraction('map_interaction', { action: 'expand' });
                if (currentTask.id === 'map') {
                  handleTaskComplete();
                }
              }}
            >
            </div>
            <div onClick={() => {
                  if (currentTask.id === 'facility') handleTaskComplete();
                }} className="section-item">
              <h3>{LANGUAGES[selectedLanguage].accessibilityFeatures}</h3>
              <ul>
                <li>{LANGUAGES[selectedLanguage].wheelchairAccessibleEntrance}</li>
                <li>{LANGUAGES[selectedLanguage].elevatorAccess}</li>
                <li>{LANGUAGES[selectedLanguage].handicapParkingAvailable}</li>
              </ul>
            </div>
          </div>
        );
        case 'about':
          return (
            <div className="section">
              <h2>{LANGUAGES[selectedLanguage].aboutUs}</h2>
              <div className="section-item">
                <h3>{LANGUAGES[selectedLanguage].volunteerOpportunities}</h3>
                <form 
                  onSubmit={(e) => {
                    e.preventDefault();
                    if (currentTask.id === 'volunteer') handleTaskComplete();
                  }}
                >
                  <input type="text" placeholder="Your Name" value="Dohn Joe" required />
                  <input type="email" placeholder="Email" value="email@gmail.com" required />
                  <button type="submit">{LANGUAGES[selectedLanguage].signUpVolunteer}</button>
                </form>
              </div>
            </div>
          );
        case 'support':
          return (
            <div className="section">
              <h2>{LANGUAGES[selectedLanguage].support}</h2>
              <div 
                className="section-item"

              >
                <h3>{LANGUAGES[selectedLanguage].makeDonation}</h3>
                <form>
                  <input type="number" placeholder="Amount ($)"  value="4.50" required />
                  <input type="text" placeholder="Card Number" value="123456789" required />
                  <button     disabled           onClick={() => {
                  if (currentTask.id === 'donation') handleTaskComplete();
                }} >{LANGUAGES[selectedLanguage].donateNow}</button>
                </form>
              </div>
            </div>
          );
        default:
        return (
          <div className="section">
            <h2>{LANGUAGES[selectedLanguage].currentTask}</h2>
            <p>{LANGUAGES[selectedLanguage].welcome}</p>
            <div className="grid">
              <div className="section-item">
                <h3>{LANGUAGES[selectedLanguage].latestNews}</h3>
                <p>{LANGUAGES[selectedLanguage].latestNewsDesc}</p>
              </div>
              <div className="section-item">
                <h3>{LANGUAGES[selectedLanguage].upcomingEvents}</h3>
                <p>{LANGUAGES[selectedLanguage].upcomingEventsDesc}</p>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="container">

      <nav className="navbar">
        <div style={{display:"flex",flexDirection:"row",width:"100%", justifyContent:"space-between"}}> 
          <h1>Community Center</h1>
          <div className="search">
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search events, classes..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              logInteraction('search', { query: e.target.value });
            }}
          />
        </div>

        </div>
        <div style={{display:"flex",flexDirection:"row",justifyContent:"space-between" ,width:"100%"}}>
        {LANGUAGES[selectedLanguage].navBar.map((item) => (
              <button style={{margin:"5px"}}
                key={item.name}
                className={activeSection === item.name ? 'active' : ''}
                onClick={() => {
                  setActiveSection(item.name);
                  logInteraction('navigation', { section: item.name });
                }}
              >
                {item.icon} {item.label}
              </button>
            ))}
          </div>
      </nav>

      <div className="task-display">
        <div className="task-header">
          <span className="alert-circle">⚠️</span>
          <h2>{LANGUAGES[selectedLanguage].currentTask}: {currentTask?.description}</h2>
        </div>
        
          <div className="hints" style={{fontSize:"20px",paddingLeft:"16px"}}>
            <ul>
              {currentTask?.hints.map((hint, index) => (
                <li key={index}>{hint}</li>
              ))}
            </ul>
          </div>
      </div>

      {renderSection()}

      {taskComplete && (
        <div className="task-completed">
          <h2>Task Completed!</h2>
          <p>Great job navigating the website!</p>
          <p>Time Taken: {((Date.now() - taskStartTime) / 1000).toFixed(2)} seconds</p>
          <button onClick={() => {nextGame()}}>Try Another Task</button>
          <button onClick={() => {
            const logBlob = new Blob(
              [JSON.stringify(interactionLog, null, 2)], 
              { type: 'application/json' }
            );
            const logUrl = URL.createObjectURL(logBlob);
            const a = document.createElement('a');
            a.href = logUrl;
            a.download = `interaction_log_${Date.now()}.json`;
            a.click();
          }}>
            Export Interaction Log
          </button>
        </div>
      )}
    </div>
  );
};

export default CommunityCenter;
